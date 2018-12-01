from math import floor
import tensorflow as tf

import pdb


# TODO: in what order should it be upscaled? (upscale -> conv OR conv -> upscale)
def vgg_small_decoder(z, args, reuse=False):
    """Decoder for VGG-11 architecture based encoder. Reconstructs next frame using given
    latent space representation

    Arguments:
        args {Arguments} -- network definition parameters
        z {np.ndarray} -- args.z_dim x 1 shaped latent space representation of first frame
    
    Keyword Arguments:
        reuse {bool} -- reuse layers in the scope (default: {False})
    
    Returns:
        recon_frame -- args.crop_width x args.crop_height x 1 shaped np.ndarray which is the
                        reconstructed next frame for the given z
    """
    with tf.variable_scope('decoder', reuse=reuse):
        # z -> 1000 fc -> 4096 fc -> 4096 fc -> flattened -> reshape as 2d to match last conv layer output of vgg11
        with tf.variable_scope('fc_layers'):
            fc1 = tf.layers.dense(inputs=z, units=1000, activation=tf.nn.relu)
            #fc2 = tf.layers.dense(inputs=fc1, units=4096, activation=tf.nn.relu)
            #fc3 = tf.layers.dense(inputs=fc2, units=4096, activation=tf.nn.relu)
            num_flattened_units = args.batch_size * floor(args.crop_height/32) * floor(args.crop_width/32) * 128
            flattened = tf.layers.dense(inputs=fc1, units=num_flattened_units)
            flattened_to_2d = tf.reshape(tensor=flattened,
                                        shape=(args.batch_size, 
                                                floor(args.crop_height/32), # 5 maxpool layers = 32x reduction
                                                floor(args.crop_width/32),  # 5 maxpool layers = 32x reduction
                                                128))                       # 64 filters in last layer
        with tf.variable_scope('deconv_layers_1'):
            # if maxpool was applied on even number shaped input, then output_shape=2xinput_shape (same padding)
            # NOTE: same padding out = in * stride
            # else output_shape=2xinput_shape+1 (valid padding)
            # NOTE: valid padding out = (in-1) * stride + filter_size
            if floor(args.crop_height/16) == flattened_to_2d.shape[1]*2:
                padding = 'same'
            else:
                padding = 'valid'

            # 2x upscale from fc output to shape of second last conv layer output in vgg11
            upscale_deconv1 = tf.layers.conv2d_transpose(inputs=flattened_to_2d, 
                                                    filters=128,
                                                    kernel_size=3, 
                                                    strides=2,
                                                    padding=padding,
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
            # no upscale transpose conv
            deconv1 = tf.layers.conv2d_transpose(inputs=upscale_deconv1, 
                                                    filters=128,
                                                    kernel_size=3, 
                                                    strides=1,
                                                    padding='same',
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
        with tf.variable_scope('deconv_layers_2'):
            # if maxpool was applied on even number shaped input, then output_shape=2xinput_shape (same padding)
            # else output_shape=2xinput_shape+1 (valid padding)
            if floor(args.crop_height/8) == deconv1.shape[1]*2:
                padding = 'same'
            else:
                padding = 'valid'

            # 2x upscale from shape of second last conv layer output to shape of third last conv 
            # layer output in vgg11
            # upscale_deconv2 = tf.layers.conv2d_transpose(inputs=deconv1, 
            #                                         filters=64,
            #                                         kernel_size=3, 
            #                                         strides=2,
            #                                         padding=padding,
            #                                         use_bias=True,
            #                                         activation=tf.nn.relu)
            # no upscale transpose conv
            deconv2 = tf.layers.conv2d_transpose(inputs=deconv1, 
                                                    filters=128,
                                                    kernel_size=3, 
                                                    strides=2,
                                                    padding='same',
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
        with tf.variable_scope('deconv_layers_3'):
            # if maxpool was applied on even number shaped input, then output_shape=2xinput_shape (same padding)
            # else output_shape=2xinput_shape+1 (valid padding)
            if floor(args.crop_height/4) == deconv2.shape[1]*2:
                padding = 'same'
            else:
                padding = 'valid'

            # 2x upscale from shape of third last conv layer output to shape of fourth last conv 
            # layer output in vgg11
            # upscale_deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, 
            #                                         filters=32,
            #                                         kernel_size=3, 
            #                                         strides=2,
            #                                         padding=padding,
            #                                         use_bias=True,
            #                                         activation=tf.nn.relu)
            # no upscale transpose conv
            deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, 
                                                    filters=64,
                                                    kernel_size=3, 
                                                    strides=2,
                                                    padding='same',
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
        with tf.variable_scope('deconv_layers_4'):
            # if maxpool was applied on even number shaped input, then output_shape=2xinput_shape (same padding)
            # else output_shape=2xinput_shape+1 (valid padding)
            if floor(args.crop_height/2) == deconv3.shape[1]*2:
                padding = 'same'
            else:
                padding = 'valid'

            # 2x upscale from shape of fourth last conv layer output to shape of fifth last conv 
            # layer output in vgg
            upscale_deconv4 = tf.layers.conv2d_transpose(inputs=deconv3, 
                                                    filters=32,
                                                    kernel_size=3, 
                                                    strides=2,
                                                    padding=padding,
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
        with tf.variable_scope('deconv_layers_5'):
            # if maxpool was applied on even number shaped input, then output_shape=2xinput_shape (same padding)
            # else output_shape=2xinput_shape+1 (valid padding)
            if args.crop_height == upscale_deconv4.shape[1]*2:
                padding = 'same'
            else:
                padding = 'valid'

            # 2x upscale from shape of fifth last conv layer output to shape of first conv 
            # layer output in vgg11
            upscale_deconv5 = tf.layers.conv2d_transpose(inputs=upscale_deconv4, 
                                                    filters=16,
                                                    kernel_size=3, 
                                                    strides=2,
                                                    padding=padding,
                                                    use_bias=True,
                                                    activation=tf.nn.relu)
            # TODO: do we want softmax here?
            # combine 64 filter outputs from previous layer into a single output (frame)
            recon_frame = tf.layers.conv2d_transpose(inputs=upscale_deconv5,
                                                        filters=1,
                                                        kernel_size=3,
                                                        strides=1,
                                                        padding='same')
    return recon_frame
