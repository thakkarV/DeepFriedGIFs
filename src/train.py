from arg_parser import parse_train_args
import encoders 
import decoders

def train(args):
    # AttributeErrors not handled
    # fail if the encoder and decoder are not found
    encoder = getattr(encoders, args.encoder)
    decoder = getattr(decoders, args.decoder)
    pass


if __name__ == "__main__":
    args = parse_train_args()
    train(args)
