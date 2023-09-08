
import torch
import torch.nn as nn

def generator_adversarial_loss(fake_preds):
    #clipped_fake_preds = torch.clamp(fake_preds, -1.0, 1.0)
    # todo do I have to apply the gen loss also across all layers?
    return -torch.mean(fake_preds)

def discriminator_adversarial_loss(real_preds, fake_preds, label_smoothing_stddev=0.1):
    smoothed_real_label = torch.normal(torch.tensor(1.0), torch.tensor(label_smoothing_stddev), size=real_preds.shape).to(real_preds.device)
    real_loss = torch.mean(nn.ReLU()(smoothed_real_label - real_preds))
    fake_loss = torch.mean(nn.ReLU()(smoothed_real_label + fake_preds))
    return real_loss + fake_loss

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation1=nn.Mish(), activation2=nn.Mish()):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding="same")

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding="same")
        self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.activation1 = activation1
        self.activation2 = activation2

        self.conv_skip = nn.Conv2d(in_channels, out_channels, 1, 1)


    def forward(self, x):
        input_ = self.conv_skip(x)
        x = self.conv(x)
        x = self.activation1(x)
        x = self.conv2(x)
        if self.activation2 is not None:
            x = self.activation2(x)
        return x + input_

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Discriminator(torch.nn.Module):
    # initializers
    def __init__(self, d=32, scale=1):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Sequential(
            ConvBlock(3, d, 3),
            nn.GroupNorm(1, d),
            #nn.InstanceNorm2d(d),
            nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
        )

        self.conv2 = torch.nn.Sequential(
            ConvBlock(d, d * 2, 3),
            nn.GroupNorm(1, d * 2),
            #nn.InstanceNorm2d(d * 2),
            nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
        )

        self.conv3 = torch.nn.Sequential(
            ConvBlock(d * 2, d * 4, 3),
            nn.GroupNorm(1, d * 4),
            #nn.InstanceNorm2d(d * 4),
            nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
        )

        self.conv4 = torch.nn.Sequential(
            ConvBlock(d * 4, d * 8, 3),
            nn.GroupNorm(1, d * 8),
            #nn.InstanceNorm2d(d * 8),
            nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
        )

        self.conv5 = torch.nn.Sequential(
            ConvBlock(d * 8, d * 16, 3),
        )

        assert scale in [1, 2, 4, 8, 16], "Scale should be 1, 2, 4, 8 or 16"

        self.scale = scale

        self.scaler = torch.nn.functional.interpolate

        self.weight_init(mean=0.0, std=0.02)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = self.scaler(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

class DiscriminatorWithFeatures(Discriminator):
    def forward_with_features(self, x):
        x = self.scaler(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        assert not torch.isnan(x1).any(), "x1 contains NaN values"
        assert not torch.isnan(x2).any(), "x2 contains NaN values"
        assert not torch.isnan(x3).any(), "x3 contains NaN values"
        assert not torch.isnan(x4).any(), "x4 contains NaN values"
        assert not torch.isnan(x5).any(), "x5 contains NaN values"

        return [x1, x2, x3,x4], x5

class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, d=32, scales=(1, 2)):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorWithFeatures(d, scale) for scale in scales
        ])
        self.scales = scales

    def forward(self, x):
        results = []
        for i, _ in enumerate(self.scales):
            results.append(self.discriminators[i].forward(x))
        return results

    def forward_with_features(self, x):
        x1x2x3x4 = []
        x5 = []
        for i, _ in enumerate(self.scales):
            x1x2x3x4_, x5_ = self.discriminators[i].forward_with_features(x)

            x1x2x3x4.append(x1x2x3x4_)
            x5.append(x5_)

        return x1x2x3x4, x5

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device='cuda'):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    fake = torch.ones_like(d_interpolates).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
def weak_feature_matching_loss(predicted_features, target_features, start_layer=2):
    num_layers = len(predicted_features)
    loss = 0
    for i in range(start_layer, num_layers):

        num_elements = torch.prod(torch.tensor(predicted_features[i].shape[1:]))
        layer_loss = nn.L1Loss()(predicted_features[i], target_features[i]) / num_elements
        assert not torch.isnan(layer_loss).any(), f"layer_loss at layer {i} contains NaN values"
        assert torch.isfinite(layer_loss).all(), f"layer_loss at layer {i} contains Inf values"
        loss += layer_loss
    return loss
