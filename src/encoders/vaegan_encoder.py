import tensorflow as tf
import pdb


def vaegan_encoder(x, args, reuse=False):
    """Encoder architecture as prescribed in VAE/GAN paper
    (https://arxiv.org/pdf/1512.09300.pdf)
    This will be used to convert a set of GIF frames into a
    representation/encoding in latent space
    Currently it encodes a single GIF frame into the latent space

    Arguments:
        x {np.ndarray} --
            args.batch_size x args.crop_width x args.crop_height x 1
            shaped batch of input frames
        args {[type]} -- arguments for training and data processing

    Keyword Arguments:
        reuse {bool} -- reuse layers in the scope (default: {False})

    Returns:
        z -- args.z_dim x 1 shaped latent space representation of input frame
    """
    # when compressing multiple frames, we need conv3d
    # we must NOT downsample along the frames axis (depth), just height and width
    # when compressing a single frame, we need conv2d
    # we downsample along height and width. there is no depth axis
    if args.window_size != 1:
        conv = tf.layers.conv3d
        downsampling_stride = (1, 2, 2)
    else:
        conv = tf.layers.conv2d
        downsampling_stride = (2, 2)

    with tf.variable_scope("encoder", reuse=reuse):
        # 5x5 64 downsampling conv, batch norm, relu
        conv1 = conv(
            inputs=x,
            filters=64,
            kernel_size=5,
            strides=downsampling_stride,
            padding='same',
            activation=None,
            use_bias=True)
        batch_norm1 = tf.layers.batch_normalization(conv1)
        relu1 = tf.nn.relu(batch_norm1)

        # 5x5 128 downsampling conv, batch norm, relu
        conv2 = conv(
            inputs=relu1,
            filters=128,
            kernel_size=5,
            strides=downsampling_stride,
            padding='same',
            activation=None,
            use_bias=True)
        batch_norm2 = tf.layers.batch_normalization(conv2)
        relu2 = tf.nn.relu(batch_norm2)

        # 5x5 256 downsampling conv, batch norm, relu
        conv3 = conv(
            inputs=relu2,
            filters=256,
            kernel_size=5,
            strides=downsampling_stride,
            padding='same',
            activation=None,
            use_bias=True)
        batch_norm3 = tf.layers.batch_normalization(conv3)
        relu3 = tf.nn.relu(batch_norm3)

        relu3_flattened = tf.layers.flatten(relu3)
        fc1 = tf.layers.dense(
            inputs=relu3_flattened,
            units=args.z_dim,
            activation=None,
            use_bias=True
        )
        batch_norm_fc = tf.layers.batch_normalization(fc1)
        z = tf.nn.relu(batch_norm_fc)

        return z
