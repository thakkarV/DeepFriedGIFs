import os.path as osp
import pickle


class SerDes(object):
    """Serializer and deserializer for compressed and uncompressed
    GIFs for use during inference. Defines the binary file format for
    embeddings so that can be written, saved and read from disk.
    """
    # we define a header for the pickle so we know we are dealing
    # with a file we wrote to disk.
    DFG_HEADER = 'DFG'

    # index into the DFG file object of various subobjects
    DFG_HEADER_IDX = 0
    DFG_WINDOW_IDX = 1
    DFG_OFFSET_IDX = 2
    DFG_CROPHI_IDX = 3
    DFG_CROPWI_IDX = 4
    DFG_COLMAP_IDX = 5
    DFG_HEADFR_IDX = 6
    DFG_COMPFR_IDX = 7
    DFG_TAILFR_IDX = 8

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
        print(path)
        with open(path, 'rb') as f:
            dfg_obj = pickle.load(f)
        return dfg_obj

    @staticmethod
    def write_compressed_gif(
        path,
        window_size,
        window_offset,
        crop_height,
        crop_width,
        colour_map,
        head_frames,
        compressed_frames,
        tail_frames):
        """Writes a compressed GIF to disk as a pickle dump.
        Arguments:
            path {str} -- Path to where the GIF will be writte
            window_size {int} -- Window size for compression
            window_offset {int} -- Window offset used for compression
            crop_height {int} -- crop height
            crop_width {int} -- crop width
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

        # construct DFG object
        dfg_obj = 9 * [None]
        dfg_obj[SerDes.DFG_HEADER_IDX] = SerDes.DFG_HEADER
        dfg_obj[SerDes.DFG_WINDOW_IDX] = window_size
        dfg_obj[SerDes.DFG_OFFSET_IDX] = window_offset
        dfg_obj[SerDes.DFG_CROPHI_IDX] = crop_height
        dfg_obj[SerDes.DFG_CROPWI_IDX] = crop_width
        dfg_obj[SerDes.DFG_COLMAP_IDX] = colour_map
        dfg_obj[SerDes.DFG_HEADFR_IDX] = head_frames
        dfg_obj[SerDes.DFG_COMPFR_IDX] = compressed_frames
        dfg_obj[SerDes.DFG_TAILFR_IDX] = tail_frames

        # write to disk
        with open(path, 'wb') as f:
            pickle.dump(dfg_obj, file=f, protocol=pickle.HIGHEST_PROTOCOL)
