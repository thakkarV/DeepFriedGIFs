import tensorflow as tf

def vae_decoder(z, args, reuse=False):
    """Decoder architecture as prescribed in VAE tutorial 
    (https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/) 
    This will be used to convert the encoding of GIF frames as input and generates the next frame as output
    Currently it merely reconstructs a single GIF frame from its encoding
    
    Arguments:
        z {np.ndarray} -- args.z_dim x 1 shaped latent space representation of frame(s)
        args {Namespace} -- arguments for training and data processing
    
    Keyword Arguments:
        reuse {bool} -- reuse layers in the scope (default: {False})
    
    Returns:
        recon_frame -- args.crop_width x args.crop_height x 1 shaped np.ndarray which is the
                        reconstructed frame for the given z
    """

    fc1 = tf.layers.dense(inputs=z, units=2048, activation="relu")
    flattened = tf.layers.dense(inputs=fc1, units=args.crop_width*args.crop_height, activation="sigmoid")
    recon_frame = tf.reshape(flattened,[args.crop_wigth, args.crop_height, 1])

    return recon_frame