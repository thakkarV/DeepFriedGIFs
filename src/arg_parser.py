import argparse

def parse_train_args():
    parser = argparse.ArgumentParser(description="Argument praser for codec training.")
    parser = add_common_args(parser)

    parser.add_argument(
        "-e", "--encoder",
        type = str,
        required = True,
        help = "Name of the encoder model to be used"
    )

    parser.add_argument(
        "-d", "--decoder",
        type = str,
        required = True,
        help = "Name of the decoder model to be used."
    )

    parser.add_argument(
        "-l", "--loss",
        type = str,
        default = None,
        required = False,
        help = "Name of the reconstruction loss to be used for training."
    )

    parser.add_argument(
        "-lr", "--learning_rate",
        type = float,
        default = 0.001,
        required = False,
        help = "Learning rate for the optimizer."
    )

    return parser.parse_args()


def parse_compress_args():
    parser = argparse.ArgumentParser(description="Argument praser for GIF compression.")
    parser = add_common_args(parser)
    parser.add_argument(
        "-e", "--encoder",
        type = str,
        required = True,
        help = "Name of the encoder model to be used"
    )

    return parser.parse_args()


def parse_decompress_args():
    parser = argparse.ArgumentParser(description="Argument praser for GIF decompression.")
    parser = add_common_args(parser)
    parser.add_argument(
        "-d", "--decoder",
        type = str,
        required = True,
        help = "Name of the decoder model to be used."
    )
    
    return parser.parse_args()


def add_common_args(parser):
    parser.add_argument(
        "-w", "--window-size",
        type = int,
        required = True,
        help = "Number of frames in the compression window."
    )

    parser.add_argument(
        "-o", "--window-offset",
        type = int,
        required = True,
        help = "Index offset of the target frame relative to the \
            first frame inthe compression window."
    )

    parser.add_argument(
        "--data-dir",
        type = str,
        default = "./data/test",
        required = True,
        help = "Path to the GIF data directory."
    )

    parser.add_argument(
        "--n-epoch",
        type = int,
        default = 1,
        required = True,
        help = "Number of epochs to train."
    )

    parser.add_argument(
        "-ch", "--crop-height",
        type = int,
        default = 100,
        required = False,
        help = "cropping width for gif."
    )

    parser.add_argument(
        "-cw", "--crop-width",
        type = int,
        default = 100,
        required = False,
        help = "cropping width for gif."
    )

    parser.add_argument(
        "-cp", "--crop_pos",
        type = str,
        default = None,
        required = False,
        help = "Crop area is UL, LL, UR, LR, or CC."
    )

    parser.add_argument(
        "-m", "--save-path",
        type = str,
        required = True,
        help = "Path to codec saved weights."
    )

    parser.add_argument(
        "--outdir",
        type = str,
        required = False,
        help = "Path to output directory and logs. \
            Defaults to <model-save-dir>/out."
    )

    return parser
