import numpy as np
from PIL import Image

def get_frames(gif):
    try:
        idx = 0
        while True:
            gif.seek(idx)
            imframe = gif.copy()
            idx += 1
            yield imframe
    except EOFError:
        pass


def test(path):
    gif = Image.open(path)
    for idx, frame in enumerate(get_frames(gif)):
        print(np.array(frame))
        
        frame.save("test{}.png".format(idx), **frame.info)
        if (idx > 2):
            break

if __name__ == "__main__":
    test("./../data/train/train.gif")
