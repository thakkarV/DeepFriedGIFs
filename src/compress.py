from arg_parser import parse_compress_args
import encoders

def compress(args):
    encoder = getattr(encoders, args.encoder)
    pass

if __name__ == "__main__":
    args = parse_compress_args()
    compress(args)
