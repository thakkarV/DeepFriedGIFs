import tensorflow as tf

def vae_encoder(x, args, reuse=False):
    """Encoder architecture as prescribed in VAE tutorial 
    (https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/) 
    This will be used to convert a set of GIF frames into a representation/encoding in latent space
    Currently it encodes a single GIF frame into the latent space
    
    Arguments:
            x {np.ndarray} -- args.batch_size x args.crop_width x args.crop_height x 1 shaped batch of input frames
            args {[type]} -- arguments for training and data processing
    
    Keyword Arguments:
            reuse {bool} -- reuse layers in the scope (default: {False})
    
    Returns:
            z -- args.z_dim x 1 shaped latent space representation of input frame
    """
    if args.window_size != 1:
        conv = tf.layers.conv3d
        downsampling_stride = (1, 2, 2)
    else:
        conv = tf.layers.conv2d
        downsampling_stride = (2, 2)
    
    with tf.variable_scope("encoder", reuse=reuse):
        # (W−F+2P)/S+1
        # 
        print("x: ", x.shape)
        conv1 = conv(
            inputs=x,
            filters=32,
            kernel_size=2,
            padding='same',
            activation='relu',
            strides=downsampling_stride
        )
        print("conv1: ", conv1.shape)
        conv2 = conv(
            inputs=conv1,
            filters=64,
            kernel_size=2,
            padding='same',
            activation='relu',
            strides=downsampling_stride
        )
        conv3 = conv(
            inputs=conv2,
            filters=128,
            kernel_size=4,
            padding='same',
            activation='relu',
            strides=downsampling_stride
        )
        conv4 = conv(
            inputs=conv3,
            filters=256,
            kernel_size=4,
            padding='same',
            activation='relu',
            strides=downsampling_stride
        )
        print("conv4: ", conv4.shape)
        flattened = tf.layers.flatten(conv4)
        hidden = tf.layers.dense(inputs=flattened, units=512, activation='relu')

        mu_z = tf.layers.dense(inputs=hidden, units=args.z_dim, activation='linear')
        log_sigma = tf.layers.dense(inputs=hidden, units=args.z_dim, activation='linear')
        print("mu_z:", mu_z.shape, "\n\n" )
        
        # fc1 = tf.layers.dense(inputs=x, units=2048, activation="relu", use_bias=True)
        # mu_z = tf.layers.dense(inputs=fc1, units=args.z_dim, activation="linear" )
        # log_sigma = tf.layers.dense(inputs=fc1, units=args.z_dim, activation="linear" )
    return mu_z, log_sigma
