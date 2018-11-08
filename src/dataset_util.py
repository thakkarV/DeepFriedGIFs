from PIL import Image
import numpy as np
import random
import fnmatch
import os

def find_files(directory, pattern):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

class Dataset(object):
    def __init__(self, datadir):
        self.datadir = datadir
        self.files   = None

    def gif_generator(self, batch_size=1):
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
            return frames, np.array(palette)


def test(path):
    dataset = Dataset(path)
    for gif, palette in dataset.gif_generator():
        print(gif)
        print(palette)
        # frame.save("test{}.png".format(idx), **frame.info)

if __name__ == "__main__":
    test("./../data/train/")
