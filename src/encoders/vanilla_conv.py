"""
NOTE: Arguments class MUST define the following
args.gif_width
args.gif_height
args.z_dim
"""
import tensorflow as tf


def vanilla_2d_conv(x, args, reuse=False):
    """VGG-11 architecture based CNN that encodes input image to latent space representation
    See ConvNet Configuration A in Table 1 of paper at https://arxiv.org/pdf/1409.1556.pdf
    
    Arguments:
        args {Arguments} -- network definition parameters
        x {np.array} -- args.gif_width x args.gif_height x 1 shaped input image
    
    Keyword Arguments:
        reuse {bool} -- reuse layers in the scope (default: {False})
    
    Returns:
        z -- args.z_dim x 1 shaped latent space representation of input image
    """
    with tf.variable_scope("encoder", reuse=reuse):
        # 64 conv -> relu -> pool
        with tf.variable_scope("conv_layers_1"):
            conv1 = tf.layers.conv2d(inputs=x, 
                                        filters=64, 
                                        kernel_size=3,
                                        strides=1,
                                        padding='same',
                                        use_bias=False)
            relu1 = tf.nn.relu(conv1)
            pool1 = tf.layers.max_pooling2d(inputs=relu1, pool_size=2, strides=2)
        # 128 conv -> relu -> pool
        with tf.variable_scope("conv_layers_2"):
            conv2 = tf.layers.conv2d(inputs=pool1, 
                                        filters=128, 
                                        kernel_size=3,
                                        strides=1,
                                        padding='same', 
                                        use_bias=False)
            relu2 = tf.nn.relu(conv2)
            pool2 = tf.layers.max_pooling2d(inputs=relu2, pool_size=2, strides=2)
        # 256 conv -> relu -> 256 conv -> relu -> pool
        with tf.variable_scope("conv_layers_3"):
            conv3_1 = tf.layers.conv2d(inputs=pool2, 
                                            filters=256, 
                                            kernel_size=3, 
                                            strides=1, 
                                            padding='same',
                                            use_bias=False)
            relu3_1 = tf.nn.relu(conv3_1)
            conv3_2 = tf.layers.conv2d(inputs=relu3_1, 
                                            filters=256, 
                                            kernel_size=3, 
                                            strides=1, 
                                            padding='same',
                                            use_bias=False)
            relu3_2 = tf.nn.relu(conv3_2)
            pool3 = tf.layers.max_pooling2d(inputs=relu3_2, pool_size=2, strides=2)
        # 512 conv -> relu -> 512 conv -> relu -> pool
        with tf.variable_scope("conv_layers_4"):
            conv4_1 = tf.layers.conv2d(inputs=pool3, 
                                            filters=512, 
                                            kernel_size=3, 
                                            strides=1, 
                                            padding='same',
                                            use_bias=False)
            relu4_1 = tf.nn.relu(conv4_1)
            conv4_2 = tf.layers.conv2d(inputs=relu4_1, 
                                            filters=512, 
                                            kernel_size=3, 
                                            strides=1, 
                                            padding='same',
                                            use_bias=False)
            relu4_2 = tf.nn.relu(conv4_2)
            pool4 = tf.layers.max_pooling2d(inputs=relu4_2, pool_size=2, strides=2)
        # 512 conv -> relu -> 512 conv -> relu -> pool -> flatten
        with tf.variable_scope("conv_layers_5"):
            conv5_1 = tf.layers.conv2d(inputs=pool4, 
                                            filters=512, 
                                            kernel_size=3, 
                                            strides=1,
                                            padding='same',
                                            use_bias=False)
            relu5_1 = tf.nn.relu(conv5_1)
            conv5_2 = tf.layers.conv2d(inputs=relu5_1, 
                                            filters=512, 
                                            kernel_size=3, 
                                            strides=1, 
                                            padding='same',
                                            use_bias=False)
            relu5_2 = tf.nn.relu(conv5_2)
            pool5 = tf.layers.max_pooling2d(inputs=relu5_2, pool_size=2, strides=2)
            pool5_flattened = tf.contrib.layers.flatten(pool5)
        # 4096 fc -> 4096 fc -> 1000 fc -> args.z_dim z
        with tf.variable_scope("fc_layers"):
            fc1 = tf.layers.dense(inputs=pool5_flattened ,units=4096)
            relu_fc1 = tf.nn.relu(fc1)
            fc2 = tf.layers.dense(inputs=relu_fc1 ,units=4096)
            relu_fc2 = tf.nn.relu(fc2)
            fc3 = tf.layers.dense(inputs=relu_fc2 ,units=1000)
            relu_fc3 = tf.nn.relu(fc3)
            z = tf.layers.dense(inputs=relu_fc3, units=args.z_dim)
    return z
