import os
import os.path as osp
import pickle as pkl

# we define a header for the pickle so we know we are dealing
# with a file we wrote to disk.
HEADER = 'DFG'


class SerDes(object):
    """Serializer and deserializer for compressed and uncompressed
    GIFs for use during inference. Defines the binary file format for
    embeddings so that can be written, saved and read from disk.
    """

    @staticmethod
    def read_compressed_gif(path):
        """Reads a compressed GIF from disk and returns the object.

        Arguments:
            path {str} -- Path to where the compressed GIF is stored

        Returns:
            obj -- Deserialized GIF as an object.
        """
        if not osp.exists(path):
            print("ERROR: Invalid path {}".format(path))
        return pkl.load(open(path, 'rb'))

    @staticmethod
    def write_compressed_gif(
        path,
        window_size,
        window_offset,
        colour_map,
        head_frames,
        compressed_frames,
        tail_frames):
        """Writes a compressed GIF to disk as a pickle dump.

        Arguments:
            path {str} -- Path to where the GIF will be writte
            window_size {int} -- Window size for compression
            window_offset {int} -- Window offset used for compression
            colour_map {None|np.ndarray} -- Nullable colour map of the GIF
            head_frames {None|np.ndarray} -- Nullable uncompressed head frames
            compressed_frames {np.ndarray} -- Compressed frames
            tail_frames {None|np.ndarray} -- Nullable uncompressed tail frames
        """
        if osp.exists(path):
            print("Warning, file at {}\n\talready exists, overrite?\ny/[N]:\t"
                  .format(path))
            ans = input()
            if (ans != 'y') and (ans != 'Y'):
                print("Aborting")
                pass

        with open(path, 'wb') as f:
            pkl.dump(
                obj=[
                    HEADER,
                    window_size,
                    window_offset,
                    colour_map,
                    head_frames,
                    compressed_frames,
                    tail_frames
                ],
                file=f,
                protocol=pkl.HIGHEST_PROTOCOL
            )
