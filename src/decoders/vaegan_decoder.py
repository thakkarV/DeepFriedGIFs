from math import floor
import tensorflow as tf


def vaegan_decoder(z, args, reuse=False):
    """Decoder architecture as prescribed in VAE/GAN paper (https://arxiv.org/pdf/1512.09300.pdf) 
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
    with tf.variable_scope("decoder", reuse=reuse):
        # number of filters in the last convulational layer of encoder
        num_last_conv_filters = 256

        # height and width to which the output of dense layer must be reshaped to feed to subsequent deconv layers
        reshape_height = floor(args.crop_height / 8)        # 3 height downsampling layers in encoder = 8x reduction
        reshape_width = floor(args.crop_width / 8)          # 3 width downsampling layers in encoder = 8x reduction

        # 8x8x256 fc, batch norm, relu
        fc1 = tf.layers.dense(inputs=z, units=num_last_conv_filters*reshape_height*reshape_height, activation=None, use_bias=True)
        batch_norm_fc = tf.layers.batch_normalization(fc1)
        relu_fc = tf.nn.relu(batch_norm_fc)

        # reshape for deconv layers
        flattened_to_2d = tf.reshape(relu_fc, shape=(args.batch_size, reshape_height, reshape_width, num_last_conv_filters))
       
        # if downsampled from even number shaped input, then output_shape=2xinput_shape (same padding)
        # NOTE: same padding out = in * stride
        # else output_shape=2xinput_shape+1 (valid padding)
        # NOTE: valid padding out = (in-1) * stride + filter_size
        if floor(args.crop_height / 4) == reshape_height*2:
            padding = 'same'
        else:
            padding = 'valid'

        # 5x5 256 upsampling conv, batch norm, relu
        conv1 = tf.layers.conv2d_transpose(inputs=flattened_to_2d,
                                            filters=256,
                                            kernel_size=5,
                                            strides=2,
                                            padding=padding,
                                            activation=None,
                                            use_bias=True)
        batch_norm1 = tf.layers.batch_normalization(conv1)
        relu1 = tf.nn.relu(batch_norm1)

        # padding for current upsampling
        if floor(args.crop_height / 2) == relu1.shape[1]*2:
            padding = 'same'
        else:
            padding = 'valid'

        # 5x5 128 upsampling conv, batch norm, relu
        conv2 = tf.layers.conv2d_transpose(inputs=relu1,
                                            filters=128,
                                            kernel_size=5,
                                            strides=2,
                                            padding=padding,
                                            activation=None,
                                            use_bias=True)
        batch_norm2 = tf.layers.batch_normalization(conv2)
        relu2 = tf.nn.relu(batch_norm2)

        # padding for current upsampling
        if floor(args.crop_height) == relu2.shape[1]*2:
            padding = 'same'
        else:
            padding = 'valid'

        # 5x5 32 upsampling conv, batch norm, relu
        conv3 = tf.layers.conv2d_transpose(inputs=relu2,
                                            filters=32,
                                            kernel_size=5,
                                            strides=2,
                                            padding=padding,
                                            activation=None,
                                            use_bias=True)
        batch_norm3 = tf.layers.batch_normalization(conv3)
        relu3 = tf.nn.relu(batch_norm3)

        # 5x5 1 conv to get reconstructed frame
        recon_frame = tf.layers.conv2d(inputs=relu3,
                                        filters=1,
                                        kernel_size=5,
                                        strides=1,
                                        padding='same',
                                        activation=tf.nn.sigmoid,
                                        use_bias=True)
        return recon_frame

