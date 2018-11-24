import os

import encoders
from dataset_util import Dataset
from arg_parser import parse_compress_args

def compress(args):
    # validate encoder
    encoder = getattr(encoders, args.encoder)

    # validate input GIF and load it
    if not os.path.exists(args.data) and not os.path.isfile(args.data):
        raise ValueError("Could not stat input GIF file.")

    frames, palette = Dataset.load_gif(args.data)
    if frames is None:
        raise ValueError(
            "Could not read GIF file at path {}".format(args.data))

    # crop if needed
    if args.crop_pos is not None:
        assert args.crop_height is not None
        assert args.crop_width  is not None

        frames = Dataset.crop_frames(
            frames,
            args.crop_pos,
            args.crop_height,
            args.crop_width
        )

if __name__ == "__main__":
    args = parse_compress_args()
    compress(args)
