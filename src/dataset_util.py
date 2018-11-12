from PIL import Image
import numpy as np
import random
import fnmatch
import os
from sys import getsizeof

def find_files(directory, pattern):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

class Dataset(object):
    def __init__(self, datadir, crop=False):
        self.datadir = datadir
        self.files   = None
        self.crop    = crop

    def gif_generator(
        self,
        batch_size=1,
        x_len=None,
        y_len=None,
        crop_pos=None):

        if (self.crop):
            assert x_len is not None
            assert y_len is not None
            assert crop_pos is not None

        if self.files is None:
            self.files = find_files(self.datadir, "*.gif")
        random.shuffle(self.files)

        gif_idx = 0
        num_gifs = len(self.files)
        while num_gifs >= gif_idx + batch_size:
            frame_batch   = []
            palette_batch = []
            for i in range(batch_size):
                frames, palette = self.get_frames(self.files[gif_idx])
                
                # now crop frames if needed
                if (self.crop):
                    frames = self.crop_frames(frames, crop_pos, x_len, y_len)

                frame_batch.append(frames)
                palette_batch.append(palette)
                gif_idx += 1
            yield frame_batch, palette_batch

    def get_frames(self, gif_file):
        gif = Image.open(gif_file)
        frames = []
        try:
            idx = 0
            while True:
                gif.seek(idx)
                frame = gif.copy()
                if idx == 0:
                    palette = frame.getpalette()
                else:
                    frame.putpalette(palette)
                idx += 1
                frames.append(np.array(frame))
        except EOFError:
            gif.close()
            return np.array(frames), np.array(palette)

    def crop_frames(self, frames, pos, x_len, y_len):
        if   pos == "UL":
            # crop and keep upper left
            raise NotImplementedError("")
            return frames
        elif pos == "LL":
            # crop and keep lower left
            raise NotImplementedError("")
            return frames
        elif pos == "UR":
            # crop and keep upper right
            raise NotImplementedError("")
            return frames
        elif pos == "LR":
            # crop and keep lower right
            raise NotImplementedError("")
            return frames
        elif pos == "CC":
            # crop and keep center
            raise NotImplementedError("")
            return frames
        else:
            raise Exception("Invalid crop position.")


def test(path):
    dataset = Dataset(path)
    for gif, palette in dataset.gif_generator():
        print(len(gif))
        print(gif[0].shape)
        print(len(palette[0]))
        print(palette[0].dtype)
        ar = np.array(palette[0], dtype=np.int8)
        print(ar.dtype)
        print(getsizeof(palette[0]))
        print(getsizeof(ar))

        # frame.save("test{}.png".format(idx), **frame.info)

if __name__ == "__main__":
    test("./../data/train/")
