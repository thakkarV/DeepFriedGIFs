from arg_parser import parse_train_args
import encoders 
import decoders
import dataset_util

def train(args):
    # AttributeErrors not handled
    # fail if the encoder and decoder are not found

    # check if crop is specfied
    if args.crop_pos == None:
        crop_flag = False
    else:
        crop_flag = True
    
    # initialize dataset
    dataset = dataset_util.Dataset(args.data_dir, crop=crop_flag)

    for gif, palette in dataset.gif_generator(gif_height=args.crop_height, gif_width=args.crop_width, crop_pos=args.crop_pos):
        print("Got batch of length ", len(gif))
        
    
    encoder = getattr(encoders, args.encoder)
    decoder = getattr(decoders, args.decoder)

    pass


if __name__ == "__main__":
    args = parse_train_args()
    
    
 

    train(args)
