import sys
import argparse
import os
import os.path
import tensorflow as tf
import helper
import warnings
import numpy as np
from distutils.version import LooseVersion
import project_tests as tests
import itertools
import labels as label_util

NUM_CLASSES_KITTI = 2
NUM_CLASSES_CITYSCAPES = 20
NUM_CATEGORIES_CITYSCAPES = 8

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    # return None, None, None, None, None

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    print(vgg_tag, vgg_path)
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, keep_prob):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    conv_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # transposed convolutions - upsample by 2
    deconv_1 = tf.layers.conv2d_transpose(conv_out, num_classes, 4, 2, 'SAME',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # deconv_1 = tf.layers.conv2d_transpose(conv_out, num_classes, 64,32, 'SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # skip connection to previous VGG layer
    skip_layer_1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    skip_conn_1 = tf.add(deconv_1, skip_layer_1)
    #skip_conn_1 = tf.layers.dropout(skip_conn_1, rate=keep_prob)

    # Upsample by 2
    deconv_2 = tf.layers.conv2d_transpose(skip_conn_1, num_classes, 4, 2, 'SAME',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    skip_layer_2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    skip_conn_2 = tf.add(deconv_2, skip_layer_2)
    # skip_conn_2 = tf.layers.dropout(skip_conn_2, rate=keep_prob)

    # Upsample by 8 (three pooling layers in VGG encoder)
    deconv_3 = tf.layers.conv2d_transpose(skip_conn_2, num_classes, 16, 8, 'SAME',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return deconv_3


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # define logits and labels
    print(num_classes)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.to_int64(tf.reshape(correct_label, [-1]))
    print(logits)
    print(correct_label, labels)
    # loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # train_op = minimise loss
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, logits, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, num_classes, num_batches_train, num_batches_dev, 
             early_stop, class_to_ignore, verbose):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_im	age: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    patience = 3  # number of times val_loss can increase before stopping
    keep_prob_stat = 0.8
    learning_rate_stat = 1e-4

    # Initialize metrics for accuracy and mean iou
    tf_label = tf.placeholder(dtype=tf.int32, shape=[None, None])
    tf_prediction = tf.placeholder(dtype=tf.int32, shape=[None, None])

    tf_iou_mask = tf.placeholder(dtype=tf.int32, shape=[None, None])
    tf_metric, tf_metric_update = tf.metrics.mean_iou(tf_label,
                                                      tf_prediction,
                                                      num_classes,
                                                      weights=tf_iou_mask,
                                                      name="metric_mean_iou")
    acc_metric, acc_update = tf.metrics.accuracy(tf_label,
                                                 tf_prediction,
                                                 name="metric_acc")
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metric_.*")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    def update_metrics(labels, ologit, class_to_ignore):
        """
        After a prediction, update the mean iou and accuracy scores.

        :param labels: Ground truth. shape = (batch_size, width, height)
        :param ologit: Predicted softmax values. shape = (batch_size, width * height)
        :return batch_accuracy, average_epoch_accuracy, avg_iou:
        """
        flattened_labels = labels.astype(int).reshape([num_images, -1])
        flattened_mask = (flattened_labels != class_to_ignore).astype(int)

        predicted_classes = np.argmax(ologit, axis=1).reshape([num_images, -1])
        feed_dict = {tf_label: flattened_labels, tf_prediction: predicted_classes, tf_iou_mask: flattened_mask}
        batch_accuracy, _ = sess.run([acc_update, tf_metric_update], feed_dict=feed_dict)
        avg_accuracy, avg_iou = sess.run([acc_metric, tf_metric])
        return batch_accuracy, avg_accuracy, avg_iou

    # Write metrics to data.txt for plotting later.
    with open("data.txt", "w") as data:
        print("Epochs\tTrain Loss\tVal Loss\tTrain Accuracy\tVal Accuracy\tTrain iou\tVal iou", file=data)
        val_loss_history = [float("inf")]
        for epoch in range(epochs):
            n = 0
            avg_loss = 0

            sess.run(running_vars_initializer)
            for images, labels in itertools.islice(get_batches_fn(batch_size, get_train=True), num_batches_train):
                num_images = images.shape[0]

                _, loss, ologit = sess.run([train_op, cross_entropy_loss, logits],
                                           feed_dict={input_image: images,
                                                      correct_label: labels,
                                                      keep_prob: keep_prob_stat,
                                                      learning_rate: learning_rate_stat})
                batch_accuracy, avg_accuracy, avg_iou = update_metrics(labels, ologit, class_to_ignore)
                avg_loss = (avg_loss * n + loss) / (n + 1)
                n += 1

                if verbose:
                    # Overwrite last line of stdout on linux. Not sure if this works on windows...
                    print(
                        "Epoch %d of %d, Batch %d: Batch loss %.4f, Batch accuracy %.4f, Avg loss: %.4f, Avg accuracy: %.4f,  Avg iou: %.4f\r" % (
                            epoch + 1, epochs, n, loss, batch_accuracy, avg_loss, avg_accuracy, avg_iou), end="")
            print(
                "\nEpoch %d of %d: Final Training loss: %.4f, Final Training accuracy: %.4f, Final Training iou: %.4f" % (
                    epoch + 1, epochs, avg_loss, avg_accuracy, avg_iou))

            n = 0
            val_loss = 0
            sess.run(running_vars_initializer)
            for images, labels in itertools.islice(get_batches_fn(batch_size, get_train=False), num_batches_dev):
                num_images = images.shape[0]
                loss, ologit = sess.run([cross_entropy_loss, logits],
                                        feed_dict={input_image: images,
                                                   correct_label: labels,
                                                   keep_prob: keep_prob_stat,
                                                   learning_rate: learning_rate_stat})
                _, val_accuracy, val_iou = update_metrics(labels, ologit, class_to_ignore)
                val_loss = (n * val_loss + loss) / (n + 1)
                n += 1
                
            print("%d\t%f\t%f\t%f\t%f\t%f\t%f" % (
                epoch + 1, avg_loss, val_loss, avg_accuracy, val_accuracy, avg_iou, val_iou), file=data)
            print("Epoch %d of %d: Val loss %.4f, Val accuracy %.4f, Val iou %.4f" % (
                epoch + 1, epochs, val_loss, val_accuracy, val_iou))

            val_loss_history.append(val_loss)
            if early_stop and helper.early_stopping(val_loss_history, patience):
                print("Early stopping. Min Val Loss:", min(val_loss_history))
                break



        compute_confusion_matrix (sess, logits, input_image, keep_prob, keep_prob_stat,
                                  learning_rate, learning_rate_stat,
                                  correct_label, get_batches_fn,
                                  cross_entropy_loss, batch_size,
                                  num_batches_train, "train")

        compute_confusion_matrix (sess, logits, input_image, keep_prob, keep_prob_stat,
                                  learning_rate, learning_rate_stat,
                                  correct_label, get_batches_fn,
                                  cross_entropy_loss, batch_size,
                                  num_batches_dev, "dev")


def compute_confusion_matrix (sess, logits, input_image, keep_prob, keep_prob_stat,
                              learning_rate, learning_rate_stat, correct_label,
                              get_batches_fn, cross_entropy_loss,
                              batch_size, num_batches_train, data_set):
    get_train = True if data_set == "train" else False

    confusion_matrix_sum = np.array (0)

    for images, labels in itertools.islice(get_batches_fn(batch_size, get_train=get_train), num_batches_train):
        num_images = images.shape[0]
        loss, ologit = sess.run([cross_entropy_loss, logits],
                                feed_dict={input_image: images,
                                           correct_label: labels,
                                           keep_prob: keep_prob_stat,
                                           learning_rate: learning_rate_stat})
                                           
        predicted_flattened = np.argmax(ologit, axis=1).reshape(-1)
        labels_flattened = labels.reshape(-1)

        m = tf.contrib.metrics.confusion_matrix (labels_flattened, predicted_flattened).eval()
        if confusion_matrix_sum.any () == False:
            confusion_matrix_sum = m
        else:
            confusion_matrix_sum += m                

        confusion_matrix_sum += tf.contrib.metrics.confusion_matrix (labels_flattened, predicted_flattened).eval()
        
    row_sum = np.sum (confusion_matrix_sum, axis = 1, keepdims = True)

    confusion_matrix_prob = confusion_matrix_sum / row_sum

    print ("For dataset " + data_set)
    print (confusion_matrix_prob)              


# tests.test_train_nn(train_nn)


def run():
    parser = argparse.ArgumentParser(description="Train and Infer FCN")
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of epochs')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--num-batches-train', default=None, type=int,
                        help='number of train batches, only adjusted for testing')
    parser.add_argument('--num-batches-dev', default=None, type=int,
                        help='number of dev batches, only adjusted for testing')

    parser.add_argument('--fast', action='store_true',
                        help='runs for 1 batch with 1 epoch')
    parser.add_argument('--data-source', default='cityscapes',
                        help='kitti or cityscapes')
    parser.add_argument('--use-classes', default=False,
                        help='If true, predict cityscape classes instead of categories')
    parser.add_argument('--scale-factor', default=4, type=int,
                        help="Scales image down on each dimension")
    parser.add_argument('--save-train', default=False,
                        help="If true, saves output train images with labels")
    parser.add_argument('--save-test', default=False,
                        help="If true, saves output test images with labels")
    parser.add_argument('--quiet', '-q', default=False, type=bool,
                        help='If true, does not print batch updates')
    parser.add_argument('--early-stop', default=True, type=bool,
                        help='If false, will not early stop')


    args = parser.parse_args()

    print("Running with arguments:")
    print(args)

    image_shape = (1024 // args.scale_factor, 2048 // args.scale_factor)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    epochs = args.epochs
    batch_size = args.batch_size
    fast_run = args.fast
    verbose = not args.quiet
    data_set = args.data_source
    num_batches_train = args.num_batches_train
    num_batches_dev = args.num_batches_dev
    use_classes = args.use_classes
    num_classes = 0
    class_to_ignore = 0
    early_stop = args.early_stop

    if data_set == "cityscapes":
        if use_classes:
            num_classes = NUM_CLASSES_CITYSCAPES
            class_to_ignore = 19
        else:
            num_classes = NUM_CATEGORIES_CITYSCAPES
    elif data_set == "kitti":
        num_classes = NUM_CLASSES_KITTI

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    label_util.init(use_classes)


    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        # Fully Convolutional Network
        last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, keep_prob)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None))
        learning_rate = tf.placeholder(dtype=tf.float32)

        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, logits, train_op, cross_entropy_loss, image_input,
                 correct_label,
                 keep_prob, learning_rate, num_classes, num_batches_train, num_batches_dev, early_stop, class_to_ignore, verbose)

        if (args.save_test):
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input, "test")
        if (args.save_train):
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input, "train")

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
