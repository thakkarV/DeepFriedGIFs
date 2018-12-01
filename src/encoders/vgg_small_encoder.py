import tensorflow as tf


def vgg_small_encoder(x, args, reuse=False):
    """VGG-11 architecture based CNN that encodes input image to latent space representation
    See ConvNet Configuration A in Table 1 of paper at https://arxiv.org/pdf/1409.1556.pdf
    
    Arguments:
        args {Arguments} -- network definition parameters
        x {np.ndarray} -- args.crop_width x args.crop_height x 1 shaped input image
    
    Keyword Arguments:
        reuse {bool} -- reuse layers in the scope (default: {False})
    
    Returns:
        z -- args.z_dim x 1 shaped latent space representation of input image
    """
    with tf.variable_scope("encoder", reuse=reuse):
        # 64 conv -> relu -> pool
        with tf.variable_scope("conv_layers_1"):
            conv1 = tf.layers.conv2d(inputs=x, 
                                        filters=16, 
                                        kernel_size=3,
                                        strides=1,
                                        padding='same',
                                        use_bias=True,
                                        activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
        # 128 conv -> relu -> pool
        with tf.variable_scope("conv_layers_2"):
            conv2 = tf.layers.conv2d(inputs=pool1, 
                                        filters=32, 
                                        kernel_size=3,
                                        strides=1,
                                        padding='same', 
                                        use_bias=True,
                                        activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
        # 256 conv -> relu -> 256 conv -> relu -> pool
        with tf.variable_scope("conv_layers_3"):
            conv3_1 = tf.layers.conv2d(inputs=pool2, 
                                            filters=64, 
                                            kernel_size=3, 
                                            strides=1, 
                                            padding='same',
                                            use_bias=True,
                                            activation=tf.nn.relu)
            conv3_2 = tf.layers.conv2d(inputs=conv3_1, 
                                            filters=64, 
                                            kernel_size=3, 
                                            strides=1, 
                                            padding='same',
                                            use_bias=True, activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=2, strides=2)
        # 512 conv -> relu -> 512 conv -> relu -> pool
        with tf.variable_scope("conv_layers_4"):
            # conv4_1 = tf.layers.conv2d(inputs=pool3, 
            #                                 filters=64, 
            #                                 kernel_size=3, 
            #                                 strides=1, 
            #                                 padding='same',
            #                                 use_bias=True, 
            #                                 activation=tf.nn.relu)
            conv4_2 = tf.layers.conv2d(inputs=pool3, 
                                            filters=128, 
                                            kernel_size=3, 
                                            strides=1, 
                                            padding='same',
                                            use_bias=True,
                                            activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=2, strides=2)
        # 512 conv -> relu -> 512 conv -> relu -> pool -> flatten
        with tf.variable_scope("conv_layers_5"):
            # conv5_1 = tf.layers.conv2d(inputs=pool4, 
            #                                 filters=64, 
            #                                 kernel_size=3, 
            #                                 strides=1,
            #                                 padding='same',
            #                                 use_bias=True,
            #                                 activation=tf.nn.relu)
            conv5_2 = tf.layers.conv2d(inputs=pool4, 
                                            filters=128, 
                                            kernel_size=3, 
                                            strides=1, 
                                            padding='same',
                                            use_bias=True,
                                            activation=tf.nn.relu)
            pool5 = tf.layers.max_pooling2d(inputs=conv5_2, pool_size=2, strides=2)
            pool5_flattened = tf.contrib.layers.flatten(pool5)
        # 4096 fc -> 4096 fc -> 1000 fc -> args.z_dim z
        with tf.variable_scope("fc_layers"):
            #fc1 = tf.layers.dense(inputs=pool5_flattened, units=4096, activation=tf.nn.relu)
            #fc2 = tf.layers.dense(inputs=fc1 ,units=4096, activation=tf.nn.relu)
            fc3 = tf.layers.dense(inputs=pool5_flattened ,units=1000, activation=tf.nn.relu)
            z = tf.layers.dense(inputs=fc3, units=args.z_dim)
    return z
