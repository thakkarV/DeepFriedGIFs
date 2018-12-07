import os
import os.path as osp
import sys


class SerDes(object):
    """Serializer and deserializer for compressed and uncompressed
    GIFs for use during inference. Defines the binary file format for
    embeddings so that can be written, saved and read from disk.
    """

    def __init__(self, window_size, window_offset):
        self.window_size = int(window_size)
        self.window_offset = int(window_offset)

    def read_compressed_gif(self, path):
        pass

    def write_compressed_gif(self, path, head_frames, compressed, tail_frames):
        pass
