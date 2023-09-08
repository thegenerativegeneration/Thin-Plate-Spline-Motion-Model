import glob
import os

import imageio
import logging

import torch

logger = logging.getLogger("TPSMM")

IMAGE_FORMATS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]

class VideoReader:
    def __init__(self, path, **kwargs):
        self.path = path
        if os.path.isdir(path):
            self.files = sorted(os.listdir(path))
            self.files = filter(lambda x: x.split(".")[-1].lower() in IMAGE_FORMATS, self.files)
            self.reader = None
        elif "%" in path:
            self.files = glob.glob(path)
            self.reader = None
        else:
            assert os.path.exists(path), f"File does not exist: {path}"
            self.reader = imageio.get_reader(path, **kwargs)

        self._index = 0


    def __enter__(self):
        if self.reader is not None:
            return self.reader.__enter__()

        return self


    def __exit__(self, type, value, traceback):
        if self.reader is not None:
            return self.reader.__exit__(type, value, traceback)

    def __del__(self):
        if self.reader is not None:
            return self.reader.__del__()

    def __iter__(self):
        if self.reader is not None:
            for frame in self.reader:
                yield frame
        else:
            for file in self.files:
                print(file)
                yield imageio.imread(os.path.join(self.path, file))

    def __len__(self):
        if self.reader is not None:
            return self.reader.__len__()
        else:
            return len(self.files)

    def __next__(self):
        if self.reader is not None:
            try:
                return self.reader.get_next_data()
            except IndexError:
                # No more frames to read
                raise StopIteration
        else:
            if self._index >= len(self.files):
                raise StopIteration
            else:
                next_file = self.files[self._index]
                self._index += 1
                return imageio.imread(os.path.join(self.path, next_file))

    def close(self):
        if self.reader is not None:
            return self.reader.close()

    def get_meta_data(self):
        if self.reader is not None:
            return self.reader.get_meta_data()
        else:
            return {}

class VideoWriter:
    def __init__(self, path, **kwargs):

        self.path = path
        self.writer = None

        print(f"VideoWriter: path={path}, kwargs={kwargs}")

        if os.path.isdir(path):
            self.path = os.path.join(path, "%05d.png")
        elif "%" in path:
            pass
        else:
            self.writer = imageio.get_writer(path, **kwargs)

        self.frame = None

    def __enter__(self):
        if self.writer is not None:
            return self.writer.__enter__()
        else:
            self.frame = 0
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            return self.writer.__exit__(exc_type, exc_val, exc_tb)
        else:
            self.frame = None
            return None

    def append_data(self, data):
        if self.writer is not None:
            print(f"VideoWriter.append_data: data.shape={data.shape}")
            return self.writer.append_data(data)
        else:
            image_path = os.path.join(self.path % self.frame)
            imageio.imwrite(image_path, data)
            self.frame += 1

    def set_meta_data(self, data):
        if self.writer is not None:
            return self.writer.set_meta_data(data)
        else:
            return None

    def close(self):
        if self.writer is not None:
            return self.writer.close()
        else:
            self.frame = None
            return None


def get_layer_type_from_key(key):
    if "conv" in key:
        return "conv"
    elif "norm" in key:
        return "norm"
    elif "occlusion" in key:
        return "occlusion"
    else:
        return None

def load_params(model, path, map_location=None, name="model",
                strict=True, find_alternative_weights=False):
    if path is None:
        return

    if os.path.isdir(path):
        path = os.path.join(path, "model.pt")

    if not os.path.exists(path):
        logger.warning(f"Could not find checkpoint at {path}")
        return

    logger.info(f"Loading checkpoint from {path}")

    checkpoint = torch.load(path, map_location=map_location)
    try:
        model.load_state_dict(checkpoint[name], strict=strict)
    except Exception as e:
        logger.error(f"Could not load model from {path}: {e}")

        if strict:
            raise e


        without_match = set()
        for k, v in model.state_dict().items():
            if k in checkpoint[name]:
                if v.shape == checkpoint[name][k].shape:
                    logger.info(f"Found direct match for {k}")
                    model.state_dict()[k].copy_(checkpoint[name][k])
                else:
                    without_match.add(k)
                    logger.warning(f'Could not find direct match for {k} in checkpoint')
            else:
                without_match.add(k)
                logger.warning(f'Could not find {k} in checkpoint')


        if find_alternative_weights:
            temp_without_match = set(without_match)

            logger.info(f"Trying to find alternative weights for {len(without_match)} keys")
            for k in temp_without_match:
                weights_type = k.split(".")[-1]
                layer_type = get_layer_type_from_key(k)

                for ckpt_k, ckpt_v in checkpoint[name].items():
                    if layer_type is None or layer_type not in ckpt_k or weights_type not in ckpt_k:
                        continue

                    if ckpt_v.shape == model.state_dict()[k].shape:
                        logger.warning(f"Found alternative weights for {k} in {ckpt_k}")
                        model.state_dict()[k].copy_(ckpt_v)
                        without_match.remove(k)
                        break

        logger.warning(f"Could not load {len(without_match)} keys from checkpoint")



