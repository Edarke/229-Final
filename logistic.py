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
LEARNING_RATE = 0.5

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    
def gen_logistic_train_data (images, labels, window_size):

    num, w, h, d = images.shape

    batch_xs = []
    batch_ys = []

    max_window_shift = window_size // 2

    for idx in range (num):
        batch_ys.append(labels[idx].reshape (-1))
        #for a single pixel
        for x in range (w):
            for y in range (h):
                if (window_size == 1):
                    f = [images[idx][x][y][0], images[idx][x][y][1], images[idx][x][y][2]]
                else:
                    f = []

                    for x_small in range (-max_window_shift, max_window_shift + 1):
                        for y_small in range (-max_window_shift, max_window_shift + 1):
                            true_x = min (max (x + x_small, 0), w - 1)
                            true_y = min (max (y + y_small, 0), h - 1)
                                
                            for color in range (3):                                
                                f.append (images[idx][true_x][true_y][color])
                    
                batch_xs.append (f)

    batch_ys = np.array(batch_ys).reshape(-1)

    assert (len (batch_xs) == len (batch_ys))
    return batch_xs, batch_ys

def train_baseline (sess, epochs, batch_size, window_size, use_extra, get_batches_fn,
                    num_classes, num_batches_train, num_batches_dev,
                    early_stop, class_to_ignore, print_confusion, verbose):
    """
    Train baseline
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    """
    patience = 3  # number of times val_loss can increase before stopping
    keep_prob_stat = 0.8
    learning_rate_stat = 1e-4

    assert (window_size % 2 ==1)
    
    #Initialize metrics for accuracy and mean iou
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

    pixels = 3*window_size**2
    #For logistic regression
    x = tf.placeholder(tf.float32, [None, pixels])
    W = tf.Variable(tf.zeros([pixels, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    logits = tf.matmul(x, W) + b

    y_in = tf.placeholder(tf.int32, [None])

    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits (labels=y_in,
                                                        logits=logits))
    
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    # Train NN using the train_nn function
    sess.run (tf.global_variables_initializer())
    
    def update_metrics(labels, ologit, class_to_ignore):
        '''
        After a prediction, update the mean iou and accuracy scores.

        :param labels: Ground truth. shape = (batch_size, width, height)
        :param ologit: Predicted softmax values. shape = (batch_size, width * height)
        :return batch_accuracy, average_epoch_accuracy, avg_iou:
        '''
        
        flattened_labels = labels.astype(int).reshape([num_images, -1])
        flattened_mask = (flattened_labels != class_to_ignore).astype(int)

        predicted_classes = np.argmax(ologit, axis=1).reshape([num_images, -1])
        feed_dict = {tf_label: flattened_labels, tf_prediction: predicted_classes, tf_iou_mask: flattened_mask}
        batch_accuracy, _ = sess.run([acc_update, tf_metric_update], feed_dict=feed_dict)
        avg_accuracy, avg_iou = sess.run([acc_metric, tf_metric])

        return batch_accuracy, avg_accuracy, avg_iou

    # Write metrics to data.txt for plotting later.
        # Write metrics to data.txt for plotting later.
    filename = "epochs_"+ str(epochs) + "batch_size" + str(batch_size) + "window_size" + str (window_size) + ".txt"
    with open(filename, "w") as data:
        
        print("Epochs\tTrain Loss\tVal Loss\tTrain Accuracy\tVal Accuracy\tTrain iou\tVal iou", file=data)
        val_loss_history = [float("inf")]
        for epoch in range(epochs):
            n = 0
            avg_loss = 0

            sess.run(running_vars_initializer)
            for images, labels in itertools.islice (get_batches_fn(batch_size, get_train=True, use_extra=use_extra),
                                                    num_batches_train):
                num_images = images.shape[0]
                batch_xs, batch_ys = gen_logistic_train_data (images, labels, window_size)
                _ , loss, ologit = sess.run ([train_step, cross_entropy, logits], feed_dict={x: batch_xs, y_in: batch_ys})
                            
                batch_accuracy, avg_accuracy, avg_iou = update_metrics(batch_ys, ologit, class_to_ignore)
                avg_loss = (avg_loss * n + loss) / (n + 1)
                
                n += 1

                if verbose:                      
                    print(
                        "Epoch %d of %d, Batch %d: Batch loss %.4f, Batch accuracy %.4f, Avg loss: %.4f, Avg accuracy: %.4f,  Avg iou: %.4f\r" % (
                            epoch + 1, epochs, n, loss, batch_accuracy, avg_loss, avg_accuracy, avg_iou), end="")
        
            print('------------------------------------------')


        if print_confusion:
            compute_confusion_matrix(sess, logits, x,
                                     y_in, get_batches_fn,
                                     cross_entropy, batch_size,
                                     num_batches_train, "train", window_size)

            compute_confusion_matrix(sess, logits, x,
                                     y_in, get_batches_fn,
                                     cross_entropy, batch_size,
                                     num_batches_dev, "dev", window_size)            

        return logits, x


def compute_confusion_matrix(sess, logits, x, y_in,
                             get_batches_fn, cross_entropy,
                             batch_size, num_batches_train, data_set, window_size):
    get_train = True if data_set == "train" else False

    confusion_matrix_sum = np.array(0)

    for images, labels in itertools.islice(get_batches_fn(batch_size, get_train=get_train), num_batches_train):
        num_images = images.shape[0]
        batch_xs, batch_ys = gen_logistic_train_data (images, labels, window_size)
        loss, ologit = sess.run ([cross_entropy, logits], feed_dict={x: batch_xs, y_in: batch_ys})                            

        predicted_flattened = np.argmax(ologit, axis=1).reshape(-1)
        labels_flattened = labels.reshape(-1)

        m = tf.contrib.metrics.confusion_matrix(labels_flattened, predicted_flattened).eval()
        if confusion_matrix_sum.any() == False:
            confusion_matrix_sum = m
        else:
            confusion_matrix_sum += m

        confusion_matrix_sum += tf.contrib.metrics.confusion_matrix(labels_flattened, predicted_flattened).eval()

    row_sum = np.sum(confusion_matrix_sum, axis=1, keepdims=True)

    confusion_matrix_prob = confusion_matrix_sum / row_sum

    print("For dataset " + data_set)
    print(confusion_matrix_prob)


def run():
    parser = argparse.ArgumentParser(description="Train and Infer FCN")
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of epochs')
    parser.add_argument('--batch-size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--num-batches-train', default=None, type=int,
                        help='number of train batches, only adjusted for testing')
    parser.add_argument('--num-batches-dev', default=None, type=int,
                        help='number of dev batches, only adjusted for testing')
    parser.add_argument('--window-size', default=1, type=int,
                        help='size of window')
    parser.add_argument('--fast', action='store_true',
                        help='runs for 1 batch with 1 epoch')
    parser.add_argument('--data-source', default='cityscapes',
                        help='kitti or cityscapes')
    parser.add_argument('--use-classes', default=False,
                        help='If true, predict cityscape classes instead of categories')
    parser.add_argument('--scale-factor', default=8, type=int,
                        help="Scales image down on each dimension")
    parser.add_argument('--save-train', action='store_true',
                        help="If ON, saves output train images with labels")
    parser.add_argument('--save-test', action='store_true',
                        help="If ON, saves output test images with labels")
    parser.add_argument('--save-val', action='store_true',
                        help="If ON, saves output val images with labels")
    parser.add_argument('--save-all-images', action='store_true',
                        help="If ON, saves all train test val images with labels")
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='If ON, does not print batch updates')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='If ON, will not early stop')
    parser.add_argument('--should-crop', action='store_true')
    parser.add_argument('--print-confusion', action='store_true')
    parser.add_argument('--use-extra', action='store_true',
                        help='If ON, will use extras provided there is data inside /data/leftImg8bits/train_extra')

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
    should_crop = args.should_crop
    print_confusion = args.print_confusion
    window_size = args.window_size

    num_classes = 0
    class_to_ignore = 0
    early_stop = not args.no_early_stop
    use_extra = args.use_extra

    if data_set == "cityscapes":
        if use_classes:
            num_classes = NUM_CLASSES_CITYSCAPES
            class_to_ignore = 19
        else:
            num_classes = NUM_CATEGORIES_CITYSCAPES
    elif data_set == "kitti":
        num_classes = NUM_CLASSES_KITTI

    label_util.init(use_classes)

    with tf.Session() as sess:

        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape,
                                                   should_crop)

        # Build NN using load_vgg, layers, and optimize function

        logits, image_input = train_baseline (sess, epochs, batch_size, window_size, use_extra, get_batches_fn,
                                              num_classes, num_batches_train, num_batches_dev,
                                              early_stop, class_to_ignore, print_confusion, verbose)

        if args.save_test or args.save_all_images:
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, None, image_input, "test")
        if args.save_train or args.save_all_images:
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, None, image_input,
                                          "train")
        if args.save_val or args.save_all_images:
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, None, image_input, "val")

            # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
