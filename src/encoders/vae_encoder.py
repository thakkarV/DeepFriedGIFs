import tensorflow as tf
def z_sample(mu_z, log_sigma):
    sampled = tf.random_normal(shape=tf.shape(mu_z))
    return mu_z + sampled*tf.exp(log_sigma/2)

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

    with tf.variable_scope("encoder", reuse=reuse):
        fc1 = tf.layers.dense(inputs=x, units=2048, activation="relu", use_bias=True)
        mu_z = tf.layers.dense(inputs=fc1, units=args.z_dim, activation="linear" )
        log_sigma = tf.layers.dense(inputs=fc1, units=args.z_dim, activation="linear" )
    return z_sample(mu_z, log_sigma)