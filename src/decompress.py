from arg_parser import parse_decompress_args
import decoders

def decompress(args):
    decoder = getattr(decoders, args.decoder)
    pass

if __name__ == "__main__":
    args = parse_decompress_args()
    decompress(args)
