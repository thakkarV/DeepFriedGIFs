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
        "-l1", "--l1-reg-strength",
        type = float,
        default = None,
        required = False,
        help = "L1 regularization strength. Use 'None' to disable."
    )

    parser.add_argument(
        "-l2", "--l2-reg-strength",
        type = float,
        default = None,
        required = False,
        help = "L2 regularization strength. Use 'None' to disable."
    )

    parser.add_argument(
        "-lr", "--learning-rate",
        type = float,
        default = 0.001,
        required = False,
        help = "Learning rate for the optimizer."
    )

    parser.add_argument(
        "-b", "--batch-size",
        type = int,
        default = 1,
        required = False,
        help = "Batch size used for training"
    )

    parser.add_argument(
        "-zd", "--z-dim",
        type = int,
        default = 100,
        required = False,
        help = "Dimensions of latent space representative vector"
    )

    parser.add_argument(
        "-it", "--input-transform",
        type = str,
        required = False,
        help = "Transform to apply to the data before input to encoder."
    )

    parser.add_argument(
        "-ot", "--output-transform",
        type = str,
        required = False,
        help = "Transform to apply to the data after output from decoder."
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

    parser.add_argument(
        "-it", "--input-transform",
        type = str,
        required = False,
        help = "Transform to apply to the data before input to encoder."
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

    parser.add_argument(
        "-ot", "--output-transform",
        type = str,
        required = False,
        help = "Transform to apply to the data after output from decoder."
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
        "-o", "--target-offset",
        type = int,
        required = True,
        help = "Index offset of the target frame relative to the \
            first frame inthe compression window."
    )

    parser.add_argument(
        "--data",
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
        "--log-interval",
        type = int,
        default = 100,
        required = False,
        help = "Number of iterations/batches after which to print train summary"
    )

    parser.add_argument(
        "-ch", "--crop-height",
        type = int,
        default = None,
        required = False,
        help = "Cropping width for GIF."
    )

    parser.add_argument(
        "-cw", "--crop-width",
        type = int,
        default = None,
        required = False,
        help = "Cropping width for GIF."
    )

    parser.add_argument(
        "-cp", "--crop-pos",
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
