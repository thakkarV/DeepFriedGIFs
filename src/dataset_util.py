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
    def __init__(self, datadir):
        self.datadir = datadir
        self.files   = None

    def gif_generator(
        self,
        batch_size=10,
        gif_height=None,
        gif_width=None,
        crop_pos=None):

        # if crop_pos is given, that means cropping is needed. therefore height and width MUST be given
        if (crop_pos != None):
            assert gif_height is not None
            assert gif_width is not None

        # if dataset has not been fetched yet, fetch it now
        if self.files is None:
            self.files = find_files(self.datadir, "*.gif")

        # shuffle every generator is restarted
        random.shuffle(self.files)

        # total number of gifs in dataset
        num_gifs = len(self.files)

        # index of gif whose frames will be passed as current batch frames
        curr_gif_idx = 0

        # ================================================== HACK SOLUTION ================================================== #
        # get frames in current gif
        while curr_gif_idx < num_gifs:
            # all frames and palette of current gif
            frames, palette = self.get_frames(self.files[curr_gif_idx])

            # now crop frames if needed
            if (crop_pos != None):
                # will return None if the gif cannot be cropped
                frames = self.crop_frames(frames, crop_pos, gif_height, gif_width)
            
            # check if cropping worked, otherwise do not add current gif to batch
            if frames.all() != None:
                num_frames_in_gif = frames.shape[0]
                
                # yield frame by frame
                for frame_idx in range(0, num_frames_in_gif):
                    # expand dims added to ensure yield output is of shape (batch_size, height, width, channels)
                    yield np.expand_dims(np.expand_dims(frames[frame_idx, :, :], axis=-1), axis=0), palette

            curr_gif_idx += 1
        # ================================================================================================================= #
        
        # ===================================================== FIXME ===================================================== #
        # i = 0
        # num_in_batch = 0
    
        # frame_batch   = []
        # palette_batch = []
        
        # while i < num_gifs:
        #     frames, palette = self.get_frames(self.files[i])
            
        #     # now crop frames if needed
        #     if (crop_pos != None):
        #         # will return None if the gif cannot be cropped
        #         frames = self.crop_frames(frames, crop_pos, gif_height, gif_width)
            
        #     # check if cropping worked, otherwise do not add current gif to batch
        #     if frames.all() != None:
        #         num_frames_in_gif = frames.shape[0]
                
        #         for frame_i in range(0,num_frames_in_gif):
        #             frame_batch.append(np.expand_dims(frames[frame_i,:,:], axis=-1))
        #             palette_batch.append(palette)
        #             num_in_batch += 1

        #             # got complete batch
        #             if num_in_batch == batch_size:
        #                 # reset number in batch and yield
        #                 num_in_batch = 0
        #                 yield frame_batch, palette_batch
                    
        #             if num_in_batch == 0:   
        #                 frame_batch   = []
        #                 palette_batch = []

        #     i+= 1

        #     # # got complete batch
        #     # if num_in_batch == batch_size:
        #     #     # reset number in batch and yield
        #     #     num_in_batch = 0
        #     #     yield frame_batch, palette_batch

        #     # # check if num_in_batch has been reset, and reset the batch arrays
        #     # if num_in_batch == 0:   
        #     #     frame_batch   = []
        #     #     palette_batch = []
        # ================================================================================================================= #

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

    def crop_frames(self, frames, pos, gif_height, gif_width):
        for frame in frames:
            num_frames, true_x, true_y = frames.shape
            if true_x < gif_height or true_y < gif_width:
                return None
            elif   pos == "UL":
                # crop and keep upper left
                return frames[:,0:gif_height,0:gif_width]
            elif pos == "LL":
                # crop and keep lower left
                return frames[:,-gif_height:,0:gif_width]
            elif pos == "UR":
                # crop and keep upper right
                return frames[:,0:gif_height,-gif_width:]
            elif pos == "LR":
                # crop and keep lower right
                return frames[:,-gif_height:,-gif_width:]
            elif pos == "CC":
                # crop and keep center

                # find center of image
                mid_x = int(true_x/2)
                mid_y = int(true_y/2)

                # find point to begin cropping at Len(cropped/2)-midpoint
                crop_beg_x = mid_x - int(gif_height/2)
                crop_beg_y = mid_y - int(gif_width/2)

                # end cropping by incrementing size
                crop_end_x = crop_beg_x + gif_height
                crop_end_y = crop_beg_y + gif_width

                return frames[:,crop_beg_x:crop_end_x,crop_beg_y:crop_end_y]
            else:
                raise Exception("Invalid crop position.")


def test(path):
    dataset = Dataset(path)
    for gif, palette in dataset.gif_generator(gif_height=100, gif_width=100, crop_pos=None):
        print("num in batch: ", len(gif))
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
