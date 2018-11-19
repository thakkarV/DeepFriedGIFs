import os
import tensorflow as tf

from encoders import vanilla_encoder
from decoders import vanilla_decoder
from arg_parser import parse_train_args
from dataset_util import Dataset
from loss import reconstruction_loss

def train(args):
    # AttributeErrors not handled
    # fail early if the encoder and decoder are not found
    # encoder = getattr(encoders, args.encoder)
    # decoder = getattr(decoders, args.decoder)
    encoder = vanilla_encoder
    decoder = vanilla_decoder
    dataset = Dataset(args.data_dir)

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
            train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss_op)

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
        summary_writer = tf.summary.FileWriter(os.path.join(args.save_path, "log"), sess.graph)
        saver = tf.train.Saver()
        
        # init graph variables
        sess.run(init_op)

        # attempt to resume previous trianing
        epoch = 0
        ckpt = tf.train.get_checkpoint_state(args.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring saved checkpoint at path {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, args.save_path)
            epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print("Restored global epoch {}".format(epoch))
        else:
            print("Strating new model training for save path {}".format(args.save_path))

        # Main training loop
        try:
            while epoch < args.n_epoch:
                # FIXME 11/18/18 18:36 - this setup takes a frame as input and reconstructs the same frame
                # this is meant only for proof of concept - need to update this after we have numbers
                for input_frames, target_frame in dataset.gif_generator(batch_size=args.batch_size,
                                                                        gif_height=args.crop_height,
                                                                        gif_width=args.crop_width,
                                                                        crop_pos=args.crop_pos):

                    loss, _, summary = sess.run([loss_op, train_op, summary_op],
                        feed_dict = {
                            X : input_frames,
                            T : input_frames    # FIXME 11/18/18 18:36
                        }
                    )

                    summary_writer.add_summary(summary)

                print("Done epoch {}".format(epoch))
                saver.save(sess, args.save_path)
        except KeyboardInterrupt:
            print("Interrupting training and saving weights")
            saver.save(sess, args.save_path)

if __name__ == "__main__":
    args = parse_train_args()
    train(args)
