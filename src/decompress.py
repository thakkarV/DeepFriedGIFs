import os.path as osp
import tensorflow as tf
import numpy as np
import transforms
from dataset_util import Dataset
from arg_parser import parse_decompress_args
from model import Model
from serdes import SerDes

import pdb


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

    # validate input GIF and load it
    if not osp.exists(args.data):
        raise ValueError("Path to input GIF does not exists.")

    if not osp.isfile(args.data):
        raise ValueError("Path to compressed file does not point to a file.")

    # name and path of the output compressed gif
    if not osp.exists(args.out_path):
        raise ValueError("Invalid path to output file.")

    if args.out_name is not None:
        out_path = osp.join(args.out_path, args.out_name + '.dfg')
    else:
        out_name = args.data.split('/')[-1].split('.')[-2]
        out_path = osp.join(args.out_path, out_name, '.dfg')

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
    decompressed_frames = np.empty(
        shape=(num_total_frames, crop_height, crop_width),
        dtype=np.int8
    )

    # write out uncompressed head and tail frames
    decompressed_frames[0:num_head_frames, :, :] = head_frames
    decompressed_frames[:-num_tail_frames, :, :] = tail_frames

    # start inference to decompress frames
    decomp_frames_raw = np.empty(
        shape=(num_comp_frames, crop_height, crop_width),
        dtype=np.float32
    )

    # load frozen graph def from pb
    with tf.gfile.GFile(osp.join(args.save_path, "decoder_graph.pb"), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph def
    with tf.Graph().as_default() as g:
        tf.import_graph_def(graph_def)

        # extract names of input and output tensors for decoder
        # input tensor (Z_placeholder) will be the first node
        # output tensor (T_hat) i.e. recon frame will be last node
        graph_ops = g.get_operations()
        Z_placeholder_tensor_name = graph_ops[0].values()[0].name
        T_hat_tensor_name = graph_ops[-1].values()[0].name

        # get tensors so that we can fetch then in inference sess
        Z_placeholder = g.get_tensor_by_name(Z_placeholder_tensor_name)
        T_hat = g.get_tensor_by_name(T_hat_tensor_name)

    with tf.Session(graph=g) as sess:
        # run all frames in a batch
        for i in range(num_comp_frames):
            decomp_frames_raw[i] = sess.run(
                T_hat,
                feed_dict={Z_placeholder: np.expand_dims(
                    compressed_frames[num_head_frames + i], axis=0)
                }
            )[:, :, 0]

    # denormalize if specified
    decomp_frames = None
    if output_transform is None:
        decomp_frames = decomp_frames_raw.astype(np.int8)
    else:
        decomp_frames = (output_transform(decomp_frames_raw)).astype(np.int8)
    assert len(decomp_frames) == num_comp_frames

    # write into frames array
    comp_start_idx = num_head_frames
    comp_end_idx = num_head_frames + num_comp_frames
    decomp_frames[comp_start_idx:comp_end_idx, :, :] = decomp_frames

    # finally serialize compressed GIF to disk
    print("Decompressed {} frames with {} head frames and {} tail frames"
          .format(num_comp_frames, num_head_frames, num_tail_frames))

    # save file to disk
    Dataset.write_gif(out_path, decomp_frames, palette)


if __name__ == "__main__":
    args = parse_decompress_args()
    decompress(args)
