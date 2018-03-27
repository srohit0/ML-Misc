# Usage: mnist_saved_model.py [--training_epochs=x] [--model_version=y] export_dir
# example-2-simple-mnist.py
import tensorflow as tf
from datetime import datetime
import time
import os
import sys
from tensorflow.examples.tutorials.mnist import input_data

# Parse command line inputs
tf.app.flags.DEFINE_integer('training_epochs', 6,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


# reset everything to rerun in jupyter
tf.reset_default_graph()

def main(_):
        # Check inputs
        if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
          print('Usage: mnist_export.py [--training_epochs=x] '
                '[--model_version=y] export_dir')
          sys.exit(-1)
        if FLAGS.training_epochs <= 0:
          print('Please specify a positive value for training epochs. ')
          sys.exit(-1)
        if FLAGS.model_version <= 0:
          print('Please specify a positive value for version number.')
          sys.exit(-1)

        # config
        batch_size = 100
        learning_rate = 0.5
        layer1_size = 200
        logs_path = 'tmp/mnistLogs' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'); #datetime.now().isoformat()

        # load mnist data set
        mnist = input_data.read_data_sets('tmp/MNIST_data', one_hot=True)

        # input images

        with tf.name_scope("weights"):
          W1 = tf.Variable(tf.truncated_normal([784, layer1_size], stddev=0.1))
          W  = tf.Variable(tf.truncated_normal([layer1_size, 10], stddev=1.0))

        with tf.name_scope('input'):
          # None -> batch size can be any size, 784 -> flattened mnist image
          #x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
          serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
          feature_configs       = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
          tf_example            = tf.parse_example(serialized_tf_example, feature_configs)
          x                     = tf.identity(tf_example['x'], name='x-input')  # use tf.identity() to assign name

          # target 10 output classes
          y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

        with tf.name_scope("biases"):
          b1 = tf.Variable(tf.zeros([layer1_size]))
          b  = tf.Variable(tf.zeros([10]))

        with tf.name_scope('hidden_layers'):
          y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

        # implement model
        with tf.name_scope("softmax"):
          ylogits = tf.matmul(y1, W) + b
          y       = tf.nn.softmax(ylogits)
          
          # Classes for model export
          values, indices    = tf.nn.top_k(y, 10)
          table              = tf.contrib.lookup.index_to_string_table_from_tensor(
                                                tf.constant([str(i) for i in range(10)]))
          prediction_classes = table.lookup(tf.to_int64(indices))


        # specify cost function
        with tf.name_scope('cross_entropy'):
          cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=ylogits, labels=y_)
          cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('train'):
          train_op = tf.train.AdamOptimizer().minimize(cross_entropy)

        with tf.name_scope('accuracy'):
          # Accuracy
          correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        train_cost_summary = tf.summary.scalar("train_cost", cross_entropy)
        train_acc_summary = tf.summary.scalar("train_accuracy", accuracy)
        test_cost_summary = tf.summary.scalar("test_cost", cross_entropy)
        test_acc_summary = tf.summary.scalar("test_accuracy", accuracy)

        # merge all summaries into a single "operation" which we can execute in a session
        # summary_op = tf.summary.merge_all()

        sess = tf.Session()
        # variables need to be initialized before we can use them
        sess.run(tf.global_variables_initializer())

        # create log writer object
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # perform training cycles
        for epoch in range(FLAGS.training_epochs):
          
          # number of batches in one epoch
          batch_count = int(mnist.train.num_examples/batch_size)

          for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # perform the operations we defined earlier on batch
            _, train_cost, train_acc, \
            _train_cost_summary, \
            _train_acc_summary = sess.run([train_op, cross_entropy, accuracy, 
                                           train_cost_summary,train_acc_summary], 
                                           feed_dict={x: batch_x, y_: batch_y})
            # write log
            writer.add_summary(_train_cost_summary, epoch * batch_count + i)
            writer.add_summary(_train_acc_summary, epoch * batch_count + i)
            if i % 100 == 0:
                # for log on test data:
                test_cost, test_acc, \
                _test_cost_summary, _test_acc_summary = sess.run([cross_entropy, accuracy,
                                                                 test_cost_summary, test_acc_summary],
                                                                 feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                # write log
                writer.add_summary(_test_cost_summary, epoch * batch_count + i)
                writer.add_summary(_test_acc_summary, epoch * batch_count + i)
                print('Epoch {0:3d}, Batch {1:3d} | Train Cost: {2:.2f} | Test Cost: {3:.2f} | Accuracy batch train: {4:.2f} | Accuracy test: {5:.2f}'
                    .format(epoch, i, train_cost, test_cost, train_acc, test_acc))
        print('Accuracy: {}'.format(accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
        print('done')

        # Export model
        export_path_base = sys.argv[-1]
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # Build the signature_def_map.
        classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
        classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
        classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

        classification_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.saved_model.signature_constants.CLASSIFY_INPUTS : classification_inputs
                },
                outputs={
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs_classes,
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES : classification_outputs_scores
                },
                method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)
        )
        tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x},
            outputs={'scores': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        builder.add_meta_graph_and_variables(sess, 
                                             [tf.saved_model.tag_constants.SERVING], 
                                             signature_def_map={
                                                 'predict_images': prediction_signature,
                                                 tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature,
                                             },
                                             legacy_init_op=legacy_init_op)
        builder.save()

        print('Done exporting!')

if __name__ == '__main__':
  tf.app.run()