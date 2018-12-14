import os.path as osp
import tensorflow as tf
import numpy as np
from dataset_util import Dataset
from arg_parser import parse_compress_args
from model import Model
from serdes import SerDes
import transforms


def compress(args):
    # TODO: we need the full graph for now. but at some point,
    # we need to decouple the encoder and decoder
    args.decoder = args.encoder.replace("encoder", "decoder")
    # TODO: these args are added here as hacks but are not really needed
    # fix at some point
    args.batch_size = 1
    args.loss = None
    args.l1_reg_strength = None
    args.l2_reg_strength = None
    args.learning_rate = 0.0005
    model = Model(args)

    input_transform = None
    if args.input_transform is not None:
        input_transform = getattr(transforms, args.input_transform)

    # validate input GIF and load it
    if not osp.exists(args.data) and not osp.isfile(args.data):
        raise ValueError("Could not stat input GIF file.")

    # name and path of the output compressed gif
    if not osp.exists(args.out_path):
        raise ValueError("Invalid path to output file.")

    if args.out_name is not None:
        out_path = osp.join(args.out_path, args.out_name + '.dfg')
    else:
        out_name = args.data.split('/')[-1].split('.')[-2]
        out_path = osp.join(args.out_path, out_name, '.dfg')

    # load GIF
    frames, palette = Dataset.load_gif(args.data)
    if frames is None:
        raise ValueError(
            "Could not read GIF file at path {}".format(args.data))

    if input_transform is not None:
        frames = input_transform(frames)

    # crop if needed
    if args.crop_pos is not None:
        assert args.crop_height > 0
        assert args.crop_width  > 0

        frames = Dataset.crop_frames(
            frames,
            args.crop_pos,
            args.crop_height,
            args.crop_width
        )

    # gather all non-compressible frames
    num_head_frames = max(0, args.target_offset)
    num_tail_frames = max(0, args.window_size - args.target_offset - 1)

    head_frames = frames[0:num_head_frames, :, :]
    tail_frames = None
    if num_tail_frames > 0:
        tail_frames = frames[-num_tail_frames:, :, :]
    num_comp_frames = frames.shape[0] - num_head_frames - num_tail_frames

    if num_comp_frames <= 0:
        print("ERROR: Cannot compress GIF at path {}".format(args.data))
        print("\tInput GIF lenght is too small for window parameters.")
    elif num_comp_frames < num_head_frames + num_tail_frames:
        print("WARNING: Number of compressed frames will be less than \
            number of uncompressed frames in the output file.")

    # init compressed frame array
    compressed_frames = np.empty(
        shape=(num_comp_frames, args.z_dim),
        dtype=np.float32
    )

    # number of frames we need to slice out of the GIF each time for inference
    # window_len = args.window_size + \
    #     max(0, args.target_offset - args.window_size)

    # start inference
    with tf.Session(graph=model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, args.save_path)

        # TODO: batch all windows together for performance
        # instead of doing this itereatively
        for i in range(num_comp_frames):
            compression_window = np.expand_dims(
                frames[i:i + args.window_size, :, :].copy(),
                axis=-1
            )

            if args.window_size > 1:
                compression_window = \
                    np.expand_dims(compression_window, axis=0)
            # if window_size is 1, then we need to shrink the input to 2D
            else:
                np.squeeze(compression_window, axis=0)

            compressed_frames[i, :] = sess.run(
                model.Z,
                feed_dict={model.X: compression_window}
            )

    # finally serialize compressed GIF to disk
    print("Compressed {} frames with {} head frames and {} tail frames"
          .format(num_comp_frames, num_head_frames, num_tail_frames))

    SerDes.write_compressed_gif(
        out_path,
        args.window_size,
        args.target_offset,
        args.crop_height,
        args.crop_width,
        palette,
        head_frames,
        compressed_frames,
        tail_frames
    )


if __name__ == "__main__":
    args = parse_compress_args()
    compress(args)
