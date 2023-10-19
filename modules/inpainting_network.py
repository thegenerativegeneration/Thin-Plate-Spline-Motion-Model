import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork

import torch


def get_kernel():
    """
    See https://setosa.io/ev/image-kernels/
    """

    k1 = torch.tensor([[0.0625, 0.125, 0.0625],
                       [0.125, 0.25, 0.125],
                       [0.0625, 0.125, 0.0625]], dtype=torch.float32)

    # Sharpening Spatial Kernel, used in paper
    k2 = torch.tensor([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]], dtype=torch.float32)

    k3 = torch.tensor([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=torch.float32)

    return k1, k2, k3


def build_sharp_blocks(layer):
    """
    Sharp Blocks
    """
    # Get number of channels in the feature
    out_channels = layer.shape[0]
    in_channels = layer.shape[1]
    # Get kernel
    _, w, _ = get_kernel()
    # Change dimension
    w = torch.unsqueeze(w, dim=0)  # add an out_channel dimension at the beginning
    # Repeat filter by out_channels times to get (out_channels, H, W)
    w = w.repeat(out_channels, 1, 1)
    # Expand dimension
    w = torch.unsqueeze(w, dim=1)  # add an in_channel dimension after out_channels
    return w


class InpaintingNetwork(nn.Module):
    """
    Inpaint the missing regions and reconstruct the Driving image.
    """
    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, multi_mask = True,
                 concat_encode=True, skip_block_type=None,
                 dropout=0.0, **kwargs):
        super(InpaintingNetwork, self).__init__()

        self.num_down_blocks = num_down_blocks
        self.multi_mask = multi_mask
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        self.concat_encode = concat_encode
        self.dropout = dropout

        down_blocks = []
        up_blocks = []
        resblock = []#
        skip_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            if skip_block_type == 'sharp':
                # depthwise conv
                skip_blocks.append(nn.Conv2d(out_features, out_features, kernel_size=(3, 3), padding=(1, 1), groups=out_features,
                                                bias=False))
                weight = build_sharp_blocks(skip_blocks[-1].weight)
                skip_blocks[-1].weight = nn.Parameter(weight, requires_grad=False)
            elif skip_block_type == 'depthwise':
                skip_blocks.append(nn.Conv2d(out_features, out_features, kernel_size=(3, 3), padding=(1, 1), groups=out_features,
                                                bias=False))
            if concat_encode:
                decoder_in_feature = out_features * 2
            else:
                decoder_in_feature = out_features

            if i==num_down_blocks-1:
                decoder_in_feature = out_features
            up_blocks.append(UpBlock2d(decoder_in_feature, in_features, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks[::-1])
        self.resblock = nn.ModuleList(resblock[::-1])
        if skip_blocks:
            self.skip_blocks = nn.ModuleList(skip_blocks)
        else:
            self.skip_blocks = None

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation,align_corners=True)

    def occlude_input(self, inp, occlusion_map):
        if not self.multi_mask:
            if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)

        out = inp * occlusion_map
        return out

    def forward(self, source_image, dense_motion):
        out = self.first(source_image) 
        encoder_map = [out]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            if self.skip_blocks:
                out = self.skip_blocks[i](out)
            encoder_map.append(out)

        output_dict = {}
        output_dict['contribution_maps'] = dense_motion['contribution_maps']
        output_dict['deformed_source'] = dense_motion['deformed_source']

        occlusion_map = dense_motion['occlusion_map']
        output_dict['occlusion_map'] = occlusion_map

        deformation = dense_motion['deformation']
        out = self.deform_input(out, deformation)
        out = self.occlude_input(out, occlusion_map[0])

        warped_encoder_maps = [out.detach()]

        for i in range(self.num_down_blocks):

            if self.dropout > 0:
                out = F.dropout2d(out, p=self.dropout, training=self.training)

            out = self.resblock[2*i](out) # e.g. 0, 2, 4, 6
            out = self.resblock[2*i+1](out) # e.g. 1, 3, 5, 7
            out = self.up_blocks[i](out) # e.g. 0, 1, 2, 3
            
            encode_i = encoder_map[-(i+2)] # e.g. -2, -3, -4, -5
            encode_i = self.deform_input(encode_i, deformation)
            
            occlusion_ind = 0
            if self.multi_mask:
                occlusion_ind = i+1
            encode_i = self.occlude_input(encode_i, occlusion_map[occlusion_ind])
            warped_encoder_maps.append(encode_i.detach())

            if(i==self.num_down_blocks-1):
                break

            if self.concat_encode:
                out = torch.cat([out, encode_i], 1)
            else:
                out = out + encode_i

        deformed_source = self.deform_input(source_image, deformation)
        output_dict["deformed"] = deformed_source
        output_dict["warped_encoder_maps"] = warped_encoder_maps

        occlusion_last = occlusion_map[-1]
        if not self.multi_mask:
            occlusion_last = F.interpolate(occlusion_last, size=out.shape[2:], mode='bilinear',align_corners=True)

        out = out * (1 - occlusion_last) + encode_i
        out = self.final(out)
        out = torch.sigmoid(out)
        out = out * (1 - occlusion_last) + deformed_source * occlusion_last
        output_dict["prediction"] = out

        return output_dict

    def get_encode(self, driver_image, occlusion_map):
        out = self.first(driver_image)
        encoder_map = []
        encoder_map.append(self.occlude_input(out.detach(), occlusion_map[-1].detach()))
        n_blocks = len(self.down_blocks)

        # len(occlusion_map) = n_blocks + 1, because of the original image size

        for i in range(n_blocks):
            out = self.down_blocks[i](out.detach())
            k = -(i+2) # reverse index
            out_mask = self.occlude_input(out.detach(), occlusion_map[k].detach())
            encoder_map.append(out_mask.detach())

        return encoder_map

