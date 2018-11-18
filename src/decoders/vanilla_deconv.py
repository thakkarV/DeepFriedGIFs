import tensorflow as tf


# TODO: in what order should it be upscaled? (upscale -> conv OR conv -> upscale)
def vanilla_deconv(z, args, reuse=False):
    """Decoder for VGG-11 architecture based encoder. Reconstructs next frame using given
    latent space representation

    Arguments:
        args {Arguments} -- network definition parameters
        z {np.ndarray} -- args.z_dim x 1 shaped latent space representation of first frame
    
    Keyword Arguments:
        reuse {bool} -- reuse layers in the scope (default: {False})
    
    Returns:
        recon_frame -- args.gif_width x args.gif_height x 1 shaped np.ndarray which is the
                        reconstructed next frame for the given z
    """
    with tf.variable_scope('decoder', reuse=reuse):
        # z -> 1000 fc -> 4096 fc -> 4096 fc -> reshape (as last conv layer output of vgg11)
        with tf.variable_scope('fc_layers'):
           fc1 = tf.layers.dense(inputs=z, units=1000, activation=tf.nn.relu)
           fc2 = tf.layers.dense(inputs=fc1, units=4096, activation=tf.nn.relu)
           fc3 = tf.layers.dense(inputs=fc2, units=4096)
           relu_fc3 = tf.nn.relu(fc3.reshape(args.batch_size, 
                                                args.gif_height/32, # 5 maxpool layers=32x reduction
                                                args.gif_widht/32,  # 5 maxpool layers=32x reduction
                                                512))               # 512 filters in last layer
        with tf.variable_scope('deconv_layers_1'):
            # 2x upscale from fc output to shape of second last conv layer output in vgg11
            upscale_deconv1 = tf.layers.conv2d_transpose(inputs=relu_fc3, 
                                                    filters=512,
                                                    kernel_size=3, 
                                                    strides=2,
                                                    padding='same',
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
            # no upscale transpose conv
            deconv1 = tf.layers.conv2d_transpose(inputs=upscale_deconv1, 
                                                    filters=512,
                                                    kernel_size=3, 
                                                    strides=1,
                                                    padding='same',
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
        with tf.variable_scope('deconv_layers_2'):
            # 2x upscale from shape of second last conv layer output to shape of third last conv 
            # layer output in vgg11
            upscale_deconv2 = tf.layers.conv2d_transpose(inputs=deconv1, 
                                                    filters=512,
                                                    kernel_size=3, 
                                                    strides=2,
                                                    padding='same',
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
            # no upscale transpose conv
            deconv2 = tf.layers.conv2d_transpose(inputs=upscale_deconv2, 
                                                    filters=512,
                                                    kernel_size=3, 
                                                    strides=1,
                                                    padding='same',
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
        with tf.variable_scope('deconv_layers_3'):
            # 2x upscale from shape of third last conv layer output to shape of fourth last conv 
            # layer output in vgg11
            upscale_deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, 
                                                    filters=256,
                                                    kernel_size=3, 
                                                    strides=2,
                                                    padding='same',
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
            # no upscale transpose conv
            deconv3 = tf.layers.conv2d_transpose(inputs=upscale_deconv3, 
                                                    filters=256,
                                                    kernel_size=3, 
                                                    strides=1,
                                                    padding='same',
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
        with tf.variable_scope('deconv_layers_4'):
            # 2x upscale from shape of fourth last conv layer output to shape of fifth last conv 
            # layer output in vgg11
            upscale_deconv4 = tf.layers.conv2d_transpose(inputs=deconv3, 
                                                    filters=128,
                                                    kernel_size=3, 
                                                    strides=2,
                                                    padding='same',
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
        with tf.variable_scope('deconv_layers_5'):
            # 2x upscale from shape of fifth last conv layer output to shape of first conv 
            # layer output in vgg11
            upscale_deconv5 = tf.layers.conv2d_transpose(inputs=upscale_deconv4, 
                                                    filters=64,
                                                    kernel_size=3, 
                                                    strides=2,
                                                    padding='same',
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
            # TODO: do we want softmax here?
            recon_frame = upscale_deconv5
    return recon_frame
