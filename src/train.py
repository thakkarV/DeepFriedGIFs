import os
import numpy as np
import tensorflow as tf

import encoders
import decoders
import transforms
from arg_parser import parse_train_args
from dataset_util import Dataset
from loss import reconstruction_loss

import pdb

def z_sample(mu_z, log_sigma):
    sampled = tf.random_normal(shape=tf.shape(mu_z))
    return mu_z + sampled*tf.exp(log_sigma/2)

def train(args):
    # AttributeErrors not handled
    # fail early if the encoder and decoder are not found
    encoder = getattr(encoders, args.encoder)
    decoder = getattr(decoders, args.decoder)
    input_transform = None
    if args.input_transform is not None:
        input_transform = getattr(transforms, args.input_transform)

    dataset = Dataset(
        args.data,
        args.batch_size,
        args.window_size,
        args.target_offset,
        args.crop_pos,
        args.crop_height,
        args.crop_width,
        input_transform
    )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # graph definition
    with tf.Graph().as_default() as g:
        # placeholders
        # target frame, number of channels is always one for GIF
        T = tf.placeholder(tf.float32, shape=(
            args.batch_size,
            args.crop_height,
            args.crop_width,
            1)
        )
        # input frame(s), this depends on network parameters
        if args.crop_pos is not None:
            # non FCN case
            if args.window_size > 1:
                X = tf.placeholder(tf.float32, shape=(
                    args.batch_size,
                    args.window_size,
                    args.crop_height,
                    args.crop_width,
                    1)
                )
            else:
                X = tf.placeholder(tf.float32, shape=(
                    args.batch_size,
                    args.crop_height,
                    args.crop_width,
                    1)
                )
        else:
            # FCN case
            if args.window_size > 1:
                X = tf.placeholder(
                    tf.float32,
                    shape=(1, args.window_size, None, None, 1)
                )
            else:
                X = tf.placeholder(
                    tf.float32,
                    shape=(1, None, None, 1)
                )
            # TODO: remove this once FCN networks have been added
            raise NotImplementedError

        # feed into networks, with their own unique name_scopes
        if args.encoder == "vae_encoder":
            mu, sigma = encoder(X, args)
            Z = z_sample(mu, sigma)
        else:
            Z = encoder(X, args)

        T_hat = decoder(Z, args)

        # calculate loss
        with tf.name_scope("loss"):
            loss_op = reconstruction_loss(T_hat, T, args.loss)

        # optimizer
        with tf.name_scope("optim"):
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            # grads = optimizer.compute_gradients(loss_op)
            train_op = optimizer.minimize(loss_op)

        # summaries
        with tf.name_scope("summary"):
            tf.summary.scalar("sumary_loss", loss_op)
            tf.summary.image("sumary_target", T)
            tf.summary.image("sumary_recon", T_hat)
            summary_op = tf.summary.merge_all()

        with tf.name_scope("init"):
            init_op = tf.global_variables_initializer()

    # graph execution
    print('starting training with learning rate = {}'.format(args.learning_rate))
    with tf.Session(graph=g) as sess:
        summary_writer = tf.summary.FileWriter(
            os.path.join(args.save_path, "log"), sess.graph)
        saver = tf.train.Saver()

        # init graph variables
        sess.run(init_op)

        # attempt to resume previous training
        epoch = 0
        ckpt = tf.train.get_checkpoint_state(args.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring saved checkpoint at path {}".format(
                ckpt.model_checkpoint_path))
            saver.restore(sess, args.save_path)
            epoch = int(ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1])
            print("Restored global epoch {}".format(epoch))
        else:
            print("Strating new model training for save path {}"
                .format(args.save_path))

        # Main training loop
        try:
            summary_global_step = 0
            while epoch < args.n_epoch:
                itr = 0
                for input_frames, target_frames, palettes in \
                    dataset.generate_training_batch():

                    loss, _, summary = sess.run(
                        [loss_op, train_op, summary_op],
                        feed_dict={
                            X: input_frames,
                            T: target_frames
                        }
                    )

                    itr += 1
                    if itr % args.log_interval == 0:
                        print("Epoch {} Itr {} loss = {}".format(epoch, itr, loss))

                        # update summaries. update global step
                        summary_writer.add_summary(
                            summary, global_step=summary_global_step)
                        summary_global_step += 1

                print("Done epoch {}".format(epoch))
                saver.save(
                    sess,
                    os.path.join(args.save_path, "model.ckpt"),
                    global_step = epoch
                )
                epoch += 1

        except KeyboardInterrupt:
            print("Interrupting training and saving weights")
            saver.save(
                sess,
                os.path.join(args.save_path, "model.ckpt"),
                global_step = epoch
            )


if __name__ == "__main__":
    args = parse_train_args()
    train(args)
