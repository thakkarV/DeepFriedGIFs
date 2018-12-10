import tensorflow as tf
import encoders
import decoders
from loss import reconstruction_loss


def z_sample(mu_z, log_sigma):
    sampled = tf.random_normal(shape=tf.shape(mu_z))
    return mu_z + sampled * tf.exp(log_sigma / 2)


class Model(object):
    def __init__(self, args):
        # AttributeErrors not handled
        # fail early if the encoder and decoder are not found
        self.encoder = getattr(encoders, args.encoder)
        self.decoder = getattr(decoders, args.decoder)

        # graph definition
        with tf.Graph().as_default() as g:
            # placeholders
            # target frame, number of channels is always one for GIF
            self.T = tf.placeholder(tf.float32, shape=(
                args.batch_size,
                args.crop_height,
                args.crop_width,
                1)
            )

            self.Z_in = tf.placeholder(tf.float32, shape=(
                args.batch_size,
                args.z_dim)
            )

            # input frame(s), this depends on network parameters
            if args.crop_pos is not None:
                # non FCN case
                if args.window_size > 1:
                    self.X = tf.placeholder(tf.float32, shape=(
                        args.batch_size,
                        args.window_size,
                        args.crop_height,
                        args.crop_width,
                        1)
                    )
                else:
                    self.X = tf.placeholder(tf.float32, shape=(
                        args.batch_size,
                        args.crop_height,
                        args.crop_width,
                        1)
                    )
            else:
                # FCN case
                if args.window_size > 1:
                    self.X = tf.placeholder(
                        tf.float32,
                        shape=(1, args.window_size, None, None, 1)
                    )
                else:
                    self.X = tf.placeholder(
                        tf.float32,
                        shape=(1, None, None, 1)
                    )
                # TODO: remove this once FCN networks have been added
                raise NotImplementedError

            # feed into networks, with their own unique name_scopes
            if args.encoder == "vae_encoder":
                mu, sigma = self.encoder(self.X, args)
                self.Z = z_sample(mu, sigma)
            else:
                mu, sigma = None, None
                self.Z = self.encoder(self.X, args)

            self.T_hat = self.decoder(self.Z, args, reuse=tf.AUTO_REUSE)
            self.decompression_op = self.decoder(
                self.Z_in, args, reuse=tf.AUTO_REUSE)

            # calculate loss
            with tf.name_scope("loss"):
                mu = mu if mu is not None else None
                sigma = sigma if sigma is not None else None
                self.loss_op = reconstruction_loss(
                    self.T_hat,
                    self.T,
                    args.loss,
                    mu, sigma,
                    args.l1_reg_strength,
                    args.l2_reg_strength
                )

            # optimizer
            with tf.name_scope("optim"):
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=args.learning_rate)
                # grads = optimizer.compute_gradients(loss_op)
                self.train_op = self.optimizer.minimize(self.loss_op)

            # summaries
            with tf.name_scope("summary"):
                tf.summary.scalar("sumary_loss", self.loss_op)
                tf.summary.image("sumary_target", self.T)
                tf.summary.image("sumary_recon", self.T_hat)
                self.summary_op = tf.summary.merge_all()

            with tf.name_scope("init"):
                self.init_op = tf.global_variables_initializer()

            self.graph = g
