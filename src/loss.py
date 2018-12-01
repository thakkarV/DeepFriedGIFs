import tensorflow as tf


def reconstruction_loss(recon, target, metric=None):
    '''Retruns the loss calculation operation for the reconstruction loss
    as measured by the input metric which defaults to squared euclidean by default.'''

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
        tf.assert_scalar(loss)
        return loss
    elif metric == "Euclidean":
        # TODO: make sure this is correct
        return tf.sqrt(tf.reduce_sum(tf.square(target - recon)))
    elif metric == "Manhattan":
        return tf.reduce_mean(target - recon)
    elif metric == "Absolute":
        return tf.reduce_sum(target - recon)

    else:
        raise NotImplementedError(
            "Metric type {} is invalid for reconstruction loss".format(metric)
        )
