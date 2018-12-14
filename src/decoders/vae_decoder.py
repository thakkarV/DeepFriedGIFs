import tensorflow as tf

def vae_decoder(z, args, reuse=False):
    """Decoder architecture as prescribed in VAE example 
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
    deconv = tf.layers.conv2d_transpose

    hidden = tf.layers.dense(inputs=z, units=2048, activation='relu')
    upsampled = tf.layers.dense(inputs=hidden, units=16384, activation='relu')
    reshaped = tf.reshape(upsampled, shape=[args.batch_size, 8, 8, 256])

    deconv1 = deconv(
        inputs=reshaped,
        filters=128,
        kernel_size=(4,4),
        padding='same',
        activation='relu',
        strides=(2,2)
    )

    print("deconv1:", deconv1.shape)
    
    deconv2 = deconv(
        inputs=deconv1,
        filters=64,
        kernel_size=(4,4),
        padding='same',
        activation='relu',
        strides=(2,2)
    )
    print("deconv2:", deconv2.shape)

    deconv3 = deconv(
        inputs=deconv2,
        filters=32,
        kernel_size=(4,4),
        padding='same',
        activation='relu',
        strides=(2,2)
    )

    recon_frame = deconv(
        inputs=deconv3,
        filters=1,
        kernel_size=(4,4),
        padding='same',
        activation='relu'
    )

    print("recon: ", recon_frame.shape)
    # fc1 = tf.layers.dense(inputs=z, units=2048, activation="relu")
    # recon_frame = tf.layers.dense(inputs=fc1, units=args.crop_width*args.crop_height, activation="sigmoid")

    return recon_frame #tf.transpose(recon_frame)
