import os
import tensorflow as tf

import encoders
import decoders
from arg_parser import parse_train_args
from dataset_util import Dataset
from loss import reconstruction_loss


def train(args):
    # AttributeErrors not handled
    # fail early if the encoder and decoder are not found
    encoder = getattr(encoders, args.encoder)
    decoder = getattr(decoders, args.decoder)
    dataset = Dataset(
        args.data,
        args.batch_size,
        args.window_size,
        args.target_offset,
        args.crop_pos,
        args.crop_height,
        args.crop_width
    )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # graph definition
    with tf.Graph().as_default() as g:
        # placeholders
        # target frame, number of channels is always one for GIF
        T = tf.placeholder(tf.float16, shape=(
            args.batch_size,
            args.crop_height,
            args.crop_width,
            1)
        )
        # input frame(s), this depends on network parameters
        if args.crop_pos is not None:
            # non FCN case
            if args.window_size > 1:
                X = tf.placeholder(tf.float16, shape=(
                    args.batch_size,
                    args.window_size,
                    args.crop_height,
                    args.crop_width,
                    1)
                )
            else:
                X = tf.placeholder(tf.float16, shape=(
                    args.batch_size,
                    args.crop_height,
                    args.crop_width,
                    1)
                )
        else:
            # FCN case
            raise NotImplementedError

        # feed into networks, with their own unique name_scopes
        Z = encoder(X, args)
        T_hat = decoder(Z, args)

        # calculate loss
        with tf.name_scope("loss"):
            loss_op = reconstruction_loss(T_hat, T, args.loss)

        # optimizer
        with tf.name_scope("optim"):
            train_op = tf.train.AdamOptimizer(
                args.learning_rate).minimize(loss_op)

        # summaries
        with tf.name_scope("summary"):
            tf.summary.scalar("loss", loss_op)
            tf.summary.tensor_summary("latent_rep", Z)
            tf.summary.image("reconstructed", T_hat)
            summary_op = tf.summary.merge_all()

        with tf.name_scope("init"):
            init_op = tf.global_variables_initializer()

    # graph execution
    with tf.Session(graph=g) as sess:
        summary_writer = tf.summary.FileWriter(
            os.path.join(args.save_path, "log"), sess.graph)
        saver = tf.train.Saver()

        # init graph variables
        sess.run(init_op)

        # attempt to resume previous trianing
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
                    summary_writer.add_summary(summary)

                    itr += 1
                    if 0 == itr % 1000:
                        print("Itr {} loss = {}".format((epoch+1)*itr, loss))

                print("Done epoch {}".format(epoch))
                saver.save(
                    sess,
                    os.path.join(args.save_path, "model.ckpt"),
                    global_step = epoch
                )

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
