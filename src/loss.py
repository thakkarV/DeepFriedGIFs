import tensorflow as tf


def reconstruction_loss(
    recon,
    target,
    metric=None,
    mu=None,
    log_sigma=None,
    l1_reg_strength=None,
    l2_reg_strength=None):
    '''Retruns the loss calculation operation for the reconstruction
    loss as measured by the input metric which defaults
    to squared euclidean by default.'''

    # average divergence between images
    if metric is None:
        loss = tf.div(
            tf.reduce_mean(
                tf.square(
                    tf.subtract(recon, target)
                )
            ),
            2,
            name="pixelwise-mse"
        )
    elif metric == "Euclidean":
        # TODO: make sure this is correct
        loss = tf.sqrt(tf.reduce_sum(tf.square(target - recon)))
    elif metric == "Manhattan":
        loss = tf.reduce_mean(target - recon)
    elif metric == "Absolute":
        loss = tf.reduce_sum(target - recon)
    elif metric == "VAE_loss":
        recon = tf.reduce_sum(
            tf.losses.sigmoid_cross_entropy(recon, target)
            
        )
        kl = tf.reduce_sum(
            tf.exp(log_sigma) + tf.square(mu) - 1 - log_sigma
            
        )
        loss = recon + 0.5*kl
    elif metric == "SSIM":
        # SSIM loss as defined in:
        # https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf
        loss = 1 - tf.reduce_mean(tf.image.ssim(target, recon, max_val=255))
    else:
        raise NotImplementedError(
            "Metric type {} is invalid for reconstruction loss".format(metric)
        )

    if l1_reg_strength is not None:
        loss += float(l1_reg_strength) * tf.add_n([tf.norm(var, ord=1) for
            var in tf.trainable_variables() if 'bias' not in var.name])

    if l2_reg_strength is not None:
        loss += float(l2_reg_strength) * tf.add_n([tf.norm(var, ord=2) for
            var in tf.trainable_variables() if 'bias' not in var.name])

    return loss
