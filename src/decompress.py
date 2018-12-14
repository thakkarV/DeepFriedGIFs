import os.path as osp
import tensorflow as tf
import numpy as np
import transforms
from dataset_util import Dataset
from arg_parser import parse_decompress_args
from model import Model
from serdes import SerDes


def nonetype_transform(inputs):
    return inputs


def decompress(args):
    # TODO: we need the full graph for now. but at some point,
    # we need to decouple the encoder and decoder
    args.encoder = args.decoder.replace("decoder", "encoder")
    # TODO: these args are added here as hacks but are not really needed
    # fix at some point
    args.batch_size = 1
    args.loss = None
    args.l1_reg_strength = None
    args.l2_reg_strength = None
    args.learning_rate = 0.0005
    model = Model(args)

    # get output transform if any
    output_transform = None
    if args.output_transform is not None:
        output_transform = getattr(transforms, args.output_transform)
    else:
        output_transform = nonetype_transform

    # validate input GIF and load it
    if not osp.exists(args.data):
        raise ValueError("Path to input GIF does not exists.")

    if not osp.isfile(args.data):
        raise ValueError("Path to compressed file does not point to a file.")

    # name and path of the output compressed gif
    if not osp.exists(args.out_path):
        raise ValueError("Invalid path to output file.")

    if args.out_name is not None:
        out_path = osp.join(args.out_path, args.out_name)
    else:
        out_name = args.data.split('/')[-1].split('.')[-2]
        out_path = osp.join(args.out_path, out_name)

    # load GIF
    dfg_obj = SerDes.read_compressed_gif(args.data)

    # validate file contents
    assert len(dfg_obj) == 9
    header = dfg_obj[0]
    if header != SerDes.DFG_HEADER:
        print("ERROR: Loaded file at path is not a valid DFG file.")
        pass

    crop_height       = dfg_obj[SerDes.DFG_CROPHI_IDX]
    crop_width        = dfg_obj[SerDes.DFG_CROPWI_IDX]
    window_size       = dfg_obj[SerDes.DFG_WINDOW_IDX]
    target_offset     = dfg_obj[SerDes.DFG_OFFSET_IDX]
    palette           = dfg_obj[SerDes.DFG_COLMAP_IDX]
    head_frames       = dfg_obj[SerDes.DFG_HEADFR_IDX]
    compressed_frames = dfg_obj[SerDes.DFG_COMPFR_IDX]
    tail_frames       = dfg_obj[SerDes.DFG_TAILFR_IDX]

    print("Loaded DFG, compressed with ws {} and offset {}"
        .format(window_size, target_offset))

    # gather all non-compressible frames
    num_head_frames = len(head_frames) if head_frames is not None else 0
    num_tail_frames = len(tail_frames) if tail_frames is not None else 0
    num_comp_frames = len(compressed_frames)
    num_total_frames = num_head_frames + num_comp_frames + num_tail_frames

    # init compressed frame array
    output_gif = np.empty(
        shape=(num_total_frames, crop_height, crop_width),
        dtype=np.uint32
    )

    # start inference to decompress frames
    decomp_frames_raw = np.empty(
        shape=(num_comp_frames, crop_height, crop_width),
        dtype=np.float32
    )
    with tf.Session(graph=model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, args.save_path)

        # run all frames in a batch
        for i in range(num_comp_frames):
            decomp_frames_raw[i] = sess.run(
                model.decompression_op,
                feed_dict={model.Z_in: np.expand_dims(
                    compressed_frames[i], axis=0)
                }
            )[0, :, :, 0]

    # denormalize if specified
    decomp_frames = (output_transform(decomp_frames_raw)).astype(np.uint8)

    if head_frames is not None:
        head_frames = (output_transform(head_frames)).astype(np.uint8)
        output_gif[0:num_head_frames, :, :] = head_frames

    if tail_frames is not None:
        tail_frames = (output_transform(tail_frames)).astype(np.uint8)
        output_gif[-num_tail_frames:, :, :] = tail_frames

    assert len(decomp_frames) == num_comp_frames

    # write into frames array
    comp_start_idx = num_head_frames
    comp_end_idx = num_head_frames + num_comp_frames
    output_gif[comp_start_idx:comp_end_idx, :, :] = decomp_frames

    # finally serialize compressed GIF to disk
    print("Decompressed {} frames with {} head frames and {} tail frames"
          .format(num_comp_frames, num_head_frames, num_tail_frames))

    # save file to disk
    Dataset.write_gif(out_path, output_gif, palette)


if __name__ == "__main__":
    args = parse_decompress_args()
    decompress(args)
