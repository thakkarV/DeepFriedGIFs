from PIL import Image
from PIL.ImagePalette import ImagePalette
import numpy as np
import random
import fnmatch
import os
import sys
import colorsys
import pdb

# index of various metadata into the state array
S_GIF_IDX_IDX = 0
S_GIF_CUR_IDX = 1
S_GIF_LEN_IDX = 2


class Dataset(object):
    """
    Dataset validation, processing and ingestion class
    for supporting training and inference.
    """

    def __init__(
            self,
            datadir,
            batch_size,
            window_size,
            target_offset,
            crop_pos=None,
            crop_height=None,
            crop_width=None,
            transform=None,
            sort_palette=False):
        """Prameterizes the dataset based on the training requirements based on
        compression window and cropping settings andvalidates input parameters.

        Arguments:
            datadir {str} -- path to data
            batch_size {int} -- batch size to be used for training
            window_size {int} -- number of frames to be used for compression
            target_offset {int} -- offset index of target frame from window

        Keyword Arguments:
            crop_pos {str} -- area to keep within the GIF (default: {None})
            crop_height {int} -- cropped height in pixels (default: {None})
            crop_width  {int} -- cropped width in pixels  (default: {None})
            transform {functor} -- applies a transform to data before yielding
                (default: {None})

        Raises:
            Exception -- in case any input parameters are not valid.
        """

        # check for dataset validity now and fail fast
        self.datadir = datadir
        self.files = Dataset.find_files(self.datadir, "*.gif")
        if self.files is None:
            raise Exception(
                "No GIF files found at dataset path: \n\t{}".format(datadir))

        # total number of gifs in the dataset
        self.num_files = len(self.files)

        # check batch_size
        if batch_size < 1:
            raise Exception("Invalid batch size")
        self.batch_size = batch_size

        if self.num_files < self.batch_size:
            raise Exception(
                "Dataset size must be at least equal to batch_size")

        # check for window parameters
        if window_size < 1 or target_offset < 0:
            raise Exception("Invalid compression window configuration.")
        self.window_size = window_size
        self.target_offset = target_offset

        # either all crop parameter must be specified or should be None
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.crop_pos = crop_pos
        crop_args = [crop_pos, crop_height, crop_width]
        if any(crop_args) and not all(crop_args):
            raise Exception(
                "Arguments for cropping need to all be None or not None")

        valid_crop_pos = ["CC", "UL", "UR", "LL", "LR", None]
        if crop_pos not in valid_crop_pos:
            raise Exception(
                "Invalid crop position. Valid options are {}"
                .format(valid_crop_pos)
            )

        # NOTE: this is for the FCN case, where a single batch
        # can have GIFs of different sizes, and we dont want to deal with that
        if crop_pos is None and batch_size > 1:
            raise Exception(
                "Batch sizes larger than 1 not supported for uncropped frames")

        if transform is not None and not callable(transform):
            raise Exception("Input transform is not callable.")
        self.transform = transform
        self.sort_palette = sort_palette

    def generate_training_batch(self):
        """Generator for the dataset during trianing. Each call retruns
        batch_size number of slices from the GIF dataset.

        Returns:
            [np.ndarray, np.ndarray, np.ndarray] --
                input compression window frames, target frame and GIF palettes
        """
        # shuffle everytime generator is restarted
        random.shuffle(self.files)

        # init booking data structures: dataset curators
        # each row stores the following metadata about
        # the GIFs in the current batch
        # curator_state[i, 0] := indoex of the GIF in self.files
        # curator_state[i, 1] := index of current window start frame
        # curator_state[i, 2] := total number of frames in the current GIF
        # NOTE: this implies that curr_gif_seek_head = max(curator_state[:, 0])
        curator_state = np.empty(shape=(self.batch_size, 3), dtype=np.int32)

        # the following two dictionaries map the index into self.files
        # to the frames and colour palette of those GIFs
        frame_dict = dict()
        palette_dict = dict()

        # init curator state for the first round
        # set initial index to -1 so that we use self.files[0]
        curator_state.fill(-1)
        next_batch_is_available = self.update_curator_state(
            curator_state, frame_dict, palette_dict)

        while next_batch_is_available:
            # get current batch
            frame_batch, target_batch, palette_batch = \
                self.extract_batch(curator_state, frame_dict, palette_dict)

            # update curator state
            next_batch_is_available = self.update_curator_state(
                curator_state, frame_dict, palette_dict)

            if self.transform is not None:
                frame_batch = self.transform(frame_batch)
                target_batch = self.transform(target_batch)
            yield frame_batch, target_batch, palette_batch

    def update_curator_state(self, curator_state, frame_dict, palette_dict):
        """handles the updates to the GIF metadata for all GIFs
        in the current batch. Loads in new GIFs if required.

        Arguments:
            curator_state {np.ndarray} -- current state of the curator array
                that needs updating
            frame_dict   {dict} -- maps the GIF index in self.files to frames
            palette_dict {dict} -- maps the GIF index in self.files to palette

        Retruns:
            boolean -- True if at least one more batch can be generated
                for the next round. False otherwise.
        """
        # first we check if all the GIFs are still valid
        # if not, we swap them out for new ones
        is_next_batch_available = True
        for i in range(self.batch_size):
            # does this GIF still have enough frames left?
            # +1 is to account for zero indexing
            required_len = curator_state[i, S_GIF_CUR_IDX] + 1 \
                + max(self.window_size, self.target_offset)

            # TODO: make sure this predicate is correct
            if required_len <= curator_state[i, S_GIF_LEN_IDX]:
                # did not run out, increment window start
                curator_state[i, S_GIF_CUR_IDX] += 1
            else:
                # ran out, load new GIF instead
                # get new GIF according to constrains
                curr_gif_idx = curator_state[i, S_GIF_IDX_IDX]
                curr_gif_seek_head = max(curator_state[:, S_GIF_IDX_IDX])
                new_gif_idx, new_frames, new_palette = \
                    self.load_gif_constrained(curr_gif_seek_head + 1)

                if new_gif_idx is not None:
                    # delete the old GIF from dicts
                    if curr_gif_idx != -1:
                        del frame_dict[curr_gif_idx]
                        del palette_dict[curr_gif_idx]

                    # update the state with the new GIF
                    curator_state[i, S_GIF_IDX_IDX] = new_gif_idx
                    curator_state[i, S_GIF_CUR_IDX] = 0
                    curator_state[i, S_GIF_LEN_IDX] = np.shape(new_frames)[0]

                    # add new frames/palette back to dicts
                    frame_dict[new_gif_idx] = new_frames
                    palette_dict[new_gif_idx] = new_palette
                else:
                    # if run out of GIF files to be picked next,
                    # we cannot abort this genrator run so relplay this one GIF
                    # for the current batch and cancel further runs afterwards.
                    # This replay will only happen once for the last batch
                    # at the end of the list of GIFs and at worst to all
                    # GIFs in the batch
                    is_next_batch_available = False

        return is_next_batch_available

    def extract_batch(self, curator_state, frame_dict, palette_dict):
        """Given the current curator state, and two dicionaries mapping
        the GIF index in curator state to its frames and colour palette,
        returns an extracted batch for a training round.

        Arguments:
            curator_state {np.ndarray} -- metadata of the batch
            frame_dict    {dict} -- {int(gif_index) : np.array(gif_frames )}
            palette_dict  {dict} -- {int(gif_index) : np.array(gif_palette)}

        Returns:
            [np.ndarray, np.ndarray, np.ndarray] --
                arrays representing the compression window input
                frames, target frame and the GIF colour palette
        """
        # TODO: sanity check this mess of colons
        # TODO: sanity check this mess of colons
        # NOTE: image dimentions will change every time in case of FCN
        # so take care of that every time we start
        # we also only look at the 0th index, because we only support
        # batch_size=1 training for FCN for now
        gif_height = np.shape(frame_dict[curator_state[0, S_GIF_IDX_IDX]])[1]
        gif_width = np.shape(frame_dict[curator_state[0, S_GIF_IDX_IDX]])[2]

        # need to account for the placeholder shape for window size
        if self.window_size == 1:
            frame_batch = np.empty(
                shape=(self.batch_size, gif_height, gif_width, 1),
                dtype=np.float32
            )
        else:
            frame_batch = np.empty(
                shape=(
                    self.batch_size,
                    self.window_size,
                    gif_height,
                    gif_width,
                    1),
                dtype=np.float32
            )

        target_batch = np.empty(
            shape=(self.batch_size, gif_height, gif_width, 1),
            dtype=np.float32
        )

        palette_batch = np.empty(
            shape=(self.batch_size, 768),
            dtype=np.uint8
        )

        # slice the input GIFs to window_size, cast to type,
        # and insert into batch np.ndarray
        for i in range(self.batch_size):
            gif_idx = curator_state[i, S_GIF_IDX_IDX]

            # frames
            frames = frame_dict[gif_idx]
            slice_idx = curator_state[i, S_GIF_CUR_IDX]
            if self.window_size == 1:
                frame_batch[i, :, :, :] = np.expand_dims(
                    frames[slice_idx, :, :].astype(np.float32),
                    axis=2
                )
            else:
                frame_batch[i, :, :, :] = np.expand_dims(
                    frames[slice_idx: slice_idx+self.window_size, :, :]
                    .astype(np.float32),
                    axis=3
                )

            # target
            target_idx = curator_state[i, S_GIF_CUR_IDX] + self.target_offset
            target_batch[i, :, :] = np.expand_dims(
                frames[target_idx, :, :].astype(np.float32),
                axis=2
            )

            # palette
            palette_batch[i, :] = palette_dict[gif_idx].astype(np.uint8)

        return frame_batch, target_batch, palette_batch

    def load_gif_constrained(self, start_file_idx):
        """Loads the next eligilble GIF for the batch from self.files
        starting from the given index. Checks for sufficient GIF length
        and GIF dimentions to make sure they are valid. Retruns None when
        dataset runs of of valid GIFs.

        Arguments:
            start_file_idx {int} -- inclusive index in self.files
                to start the search from

        Retruns:
            [int, np.ndarray, np.ndarray] --
                GIF file index, cropped frames and colour palette
        """
        # TODO: make sure the edge cases are correct here
        i = start_file_idx
        while i < self.num_files:
            frames, palette = None, None

            # NOTE: this is for the pesky case of palette == None
            while palette is None and i < self.num_files:
                frames, palette = Dataset.load_gif(self.files[i])
                if (self.crop_pos is not None and (
                        frames[0].shape[0] < self.crop_height
                        or frames[0].shape[1] < self.crop_width)
                    ):
                    palette = None
                i += 1

            if palette is None:
                return None, None, None

            required_len = max(self.window_size, self.target_offset)
            gif_len, gif_height, git_width = np.shape(frames)
            if gif_len >= required_len:
                # no croping needed in FCN training, crop otherwise
                if (self.crop_pos is not None):
                    frames = Dataset.crop_frames(
                        frames,
                        self.crop_pos,
                        self.crop_height,
                        self.crop_width)

                if self.sort_palette:
                    Dataset.sort_palette(frames, palette)
                return i, frames, palette
            else:
                # if these conditions are not met, then the GIF
                # is not large enough for long enough for trianing
                del frames
                del palette
                i += 1

        # ran out of data
        return None, None, None

    @staticmethod
    def crop_frames(frames, pos, crop_height, crop_width):
        """Crops a loaded GIF according to cropping params

        Arguments:
            frames {np.ndarray} -- input frames that are to be cropped
            pos {str} -- Position of area to keep within the GIF
            crop_height {int} -- Cropped height in pixels
            crop_width  {int} -- Cropped width in pixels

        Returns:
            np.ndarray -- cropped frames
        """
        _, true_x, true_y = frames.shape
        if true_x < crop_height or true_y < crop_width:
            return None
        elif pos == "UL":
            # crop and keep upper left
            return frames[:, 0:crop_height, 0:crop_width]
        elif pos == "LL":
            # crop and keep lower left
            return frames[:, -crop_height:, 0:crop_width]
        elif pos == "UR":
            # crop and keep upper right
            return frames[:, 0:crop_height, -crop_width:]
        elif pos == "LR":
            # crop and keep lower right
            return frames[:, -crop_height:, -crop_width:]
        else:
            # crop and keep center
            # find center of image
            mid_x = int(true_x/2)
            mid_y = int(true_y/2)

            # find point to begin cropping at Len(cropped/2)-midpoint
            crop_beg_x = mid_x - int(crop_height/2)
            crop_beg_y = mid_y - int(crop_width/2)

            # end cropping by incrementing size
            crop_end_x = crop_beg_x + crop_height
            crop_end_y = crop_beg_y + crop_width

            return frames[:, crop_beg_x:crop_end_x, crop_beg_y:crop_end_y]

    @staticmethod
    def find_files(directory, pattern):
        '''Recursively finds all files matching the pattern.'''
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
        return files

    @staticmethod
    def load_gif(gif_file):
        """Reads a single GIF and returns the frames and the colour
        palette as numpy arrays.

        Arguments:
            gif_file {str} -- path to GIF file

        Returns:
            [np.ndarray, np.ndarray] -- raw frames and colour palette
        """

        # deal with corrupted or otherwise unreadable GIF files
        try:
            gif = Image.open(gif_file)
        except Exception as e:
            print("Could not open GIF file at path {}".format(gif_file),
                  file=sys.stderr)
            print("\tGot exception while trying to open: e=\n\t{}".format(e),
                  file=sys.stderr)
            return None, None

        frames = []
        try:
            idx = 0
            while True:
                gif.seek(idx)
                frame = gif.copy()
                if idx == 0:
                    palette = frame.getpalette()
                    # TODO: deal with this None palette case
                    if palette is None:
                        return None, None
                else:
                    frame.putpalette(palette)
                idx += 1
                frames.append(np.array(frame))

        except EOFError:
            gif.close()
            frames = np.array(frames, dtype=np.float32)
            palette = np.array(palette, dtype=np.uint8)
            return frames, palette

    @staticmethod
    def write_gif(path, frames, palette):
        """Writes a GIF file to disk by generating a new PIL GIF
        from the input frames and color map.

        Arguments:
            path {str} -- full path to output gif including file_name.gif
            frames {np.ndarray} -- Frames of the gif in dtype=np.uint8
            palette {np.ndarray} -- Colour palette of the gif in dtype=np.uint8
        """
        file = open(path, 'wb')
        # NOTE: mode 'P' means:
        # "8-bit pixels, mapped to any other mode using a color palette"
        first_img = Image.fromarray(frames[0], mode='P')
        other_imgs = []

        # now we have to generate each frame as PIL image first
        for i in range(1, frames.shape[0]):
            other_imgs.append(Image.fromarray(frames[i], mode='P'))

        pil_palette = ImagePalette(
            mode='P', palette=bytearray(palette), size=len(palette))
        first_img.save(
            file,
            format="GIF",
            save_all=True,
            append_images=other_imgs,
            palette=pil_palette,
            loop=0
        )

        file.close()

    @staticmethod
    def sort_palette(frames, palette):
        """Sorts the colour palette to have 'similar' colours next to
        each other according to their HSV value and remaps the input frames
        and palette according to the new colour order.

        Arguments:
            frames {np.ndarray} -- frames of the GIF
            palette {np.ndarray} -- colour palette

        Returns:
            {np.ndarray, np.ndarray} -- sorted palette and remapped frames
        """
        colours = []
        for i in range(int(len(palette) / 3)):
            colours.append(
                [palette[i * 3], palette[(i * 3) + 1], palette[(i * 3) + 2]]
            )

        sorted_idx_col_list = sorted(
            enumerate(colours),
            key=lambda idx_col: colorsys.rgb_to_hsv(*idx_col[1])
        )

        idx_map = len(sorted_idx_col_list) * [None]
        for i in range(len(sorted_idx_col_list)):
            idx_map[sorted_idx_col_list[i][0]] = i

        # replace colours in palette
        for i in range(int(len(palette) / 3)):
            palette[(i * 3) + 0] = sorted_idx_col_list[i][1][0]
            palette[(i * 3) + 1] = sorted_idx_col_list[i][1][1]
            palette[(i * 3) + 2] = sorted_idx_col_list[i][1][2]

        # replace indexes in frames
        for frame in frames:
            for i in range(frame.shape[0]):
                for j in range(frame.shape[1]):
                    frame[i, j] = idx_map[int(frame[i, j])]

        return frames, palette


if __name__ == "__main__":

    # test FCN case with bs = 1 -- no cropping
    dataset = Dataset(
        "./../data/train/",
        batch_size=1,
        window_size=3,
        target_offset=1)
    for i, batch in enumerate(dataset.generate_training_batch()):
        print(i)
        print(np.shape(batch[0]))
        print(np.shape(batch[1]))
        print(np.shape(batch[2]))
        if i > 512:
            break

    # test non FCN case -- cropping
    from transforms import normalize
    dataset = Dataset(
        "./../data/train/",
        batch_size=64,
        window_size=3,
        target_offset=1,
        crop_pos="CC",
        crop_height=64,
        crop_width=64,
        transform=normalize)
    for i, batch in enumerate(dataset.generate_training_batch()):
        print(i)
        pdb.set_trace()
        print(np.shape(batch[0]))
        print(np.shape(batch[1]))
        print(np.shape(batch[2]))
        if i > 2048:
            break
