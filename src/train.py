import os
import tensorflow as tf
from arg_parser import parse_train_args
from dataset_util import Dataset
from model import Model
import transforms


def train(args):
    model = Model(args)

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

    # graph execution
    print('starting training with learning rate = {}'
          .format(args.learning_rate))
    with tf.Session(graph=model.graph) as sess:
        summary_writer = tf.summary.FileWriter(
            os.path.join(args.save_path, "log"), sess.graph)
        saver = tf.train.Saver()

        # init graph variables
        sess.run(model.init_op)

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
                        [model.loss_op, model.train_op, model.summary_op],
                        feed_dict={
                            model.X: input_frames,
                            model.T: target_frames
                        }
                    )

                    itr += 1
                    if itr % args.log_interval == 0:
                        print("Epoch {} Itr {} loss = {}"
                              .format(epoch, itr, loss))

                        # update summaries. update global step
                        summary_writer.add_summary(
                            summary, global_step=summary_global_step)
                        summary_global_step += 1

                print("Done epoch {}".format(epoch))
                saver.save(
                    sess,
                    os.path.join(args.save_path, "model.ckpt"),
                    global_step=epoch
                )
                epoch += 1

        except KeyboardInterrupt:
            print("Interrupting training and saving weights")
            saver.save(
                sess,
                os.path.join(args.save_path, "model.ckpt"),
                global_step=epoch
            )

    # get encoder and decoder subgraphs for inference
    encoder_graph_def = tf.graph_util.extract_sub_graph(model.graph.as_graph_def(),
                                                        dest_nodes=[model.Z.name[:-2]])
    encoder_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                        encoder_graph_def,
                                                                        [model.Z.name[:-2]])
    decoder_graph_def = tf.graph_util.extract_sub_graph(model.graph.as_graph_def(),
                                                        dest_nodes=[model.decompression_op.name[:-2]])
    decoder_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                        decoder_graph_def,
                                                                        [model.decompression_op.name[:-2]])

    # save frozen graphs for inference
    with tf.gfile.GFile(os.path.join(args.save_path, "encoder_graph.pb"), "wb") as f:
        f.write(encoder_graph_def.SerializeToString())
    with tf.gfile.GFile(os.path.join(args.save_path, "decoder_graph.pb"), "wb") as f:
        f.write(decoder_graph_def.SerializeToString())


if __name__ == "__main__":
    args = parse_train_args()
    train(args)
