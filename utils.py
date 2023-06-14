import glob
import os

import imageio
import logging

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


