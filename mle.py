import sys
import argparse
import os
import os.path
import scipy.misc

import scipy
import tensorflow as tf
import time

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


def layers(input, num_classes):
    return tf.layers.conv2d(
        input,
        num_classes,
        kernel_size=1,
        padding="SAME",
    )


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


def train_nn(sess, epochs, batch_size, use_extra, get_batches_fn, logits, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, num_classes, num_batches_train, num_batches_dev,
             early_stop, class_to_ignore, print_confusion, verbose):
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

    def update_metrics(labels, preds, class_to_ignore):
        """
        After a prediction, update the mean iou and accuracy scores.

        :param labels: Ground truth. shape = (batch_size, width, height)
        :param ologit: Predicted softmax values. shape = (batch_size, width * height)
        :return batch_accuracy, average_epoch_accuracy, avg_iou:
        """
        print(labels.astype(int).reshape([4, -1]))
        flattened_labels = labels.astype(int).reshape([num_images, -1])
        flattened_mask = (flattened_labels != class_to_ignore).astype(int)

        # predicted_classes = np.argmax(ologit, axis=1).reshape([num_images, -1])
        preds = np.reshape(preds, [num_images, -1])
        feed_dict = {tf_label: flattened_labels, tf_prediction: preds, tf_iou_mask: flattened_mask}
        sess.run([tf_metric_update], feed_dict=feed_dict)
        avg_iou = sess.run([tf_metric])
        return 0, 0, avg_iou

    # Write metrics to data.txt for plotting later.
    class_count = np.zeros(num_classes)
    bucket_size = 16
    res = 256//bucket_size

    color_count = np.ones([res, res, res, num_classes])
    output_dir = os.path.join("runs", str(time.time()) + "/" + "evan")
    os.makedirs(output_dir)

    with open("data.txt", "w") as data:
        print("Epochs\tTrain Loss\tVal Loss\tTrain Accuracy\tVal Accuracy\tTrain iou\tVal iou", file=data)
        for epoch in range(2):
            n = 0

            sess.run(running_vars_initializer)
            for images, labels in itertools.islice(get_batches_fn(batch_size, get_train=True, use_extra=use_extra),
                                                   num_batches_train):
                num_images = images.shape[0]


                preds = []
                for i, img in enumerate(images):
                    pred = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)

                    for y in range((img.shape[0])):
                        for x in range((img.shape[1])):
                            color = np.zeros(3, dtype=np.int)
                            color = (img[y][x] + bucket_size//2) // bucket_size
                            pred[y][x] = np.argmax(color_count[color[0], color[1], color[2]])
                    preds.append(pred)

                    for y in range((img.shape[0])):
                        for x in range((img.shape[1])):
                            color = np.zeros(3, dtype=np.int)
                            color = (img[y][x] + bucket_size//2) // bucket_size
                            label = int(labels[i][y][x])
                            class_count[label] += 1
                            color_count[color[0], color[1], color[2], label] += 1
                    if i == 0:
                        mask2 = label_util.get_color_matrix(pred)
                        mask = scipy.misc.toimage(mask2, mode="RGBA")
                        street_im = scipy.misc.toimage(img)
                        street_im.paste(mask, box=None, mask=mask)
                        scipy.misc.imsave(os.path.join(output_dir, "example.png"), street_im)

                batch_accuracy, avg_accuracy, avg_iou = update_metrics(labels, preds, class_to_ignore)
                n += 1
                print("IOU", avg_iou)
            print("End Epoch")




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
        class_to_ignore = -1
        num_classes = NUM_CLASSES_KITTI

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    label_util.init(use_classes)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape,
                                                   should_crop)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        image_input, _, _, _, _ = load_vgg(sess, vgg_path)
        # Fully Convolutional Network
        last_layer = layers(image_input, num_classes)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None))
        learning_rate = tf.placeholder(dtype=tf.float32)

        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, use_extra, get_batches_fn, logits, train_op, cross_entropy_loss, image_input,
                 correct_label,
                 1., learning_rate, num_classes, num_batches_train, num_batches_dev, early_stop, class_to_ignore,
                 print_confusion, verbose)

        if args.save_test or args.save_all_images:
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, 1., image_input, "test")
        if args.save_train or args.save_all_images:
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, 1., image_input,
                                          "train")
        if args.save_val or args.save_all_images:
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, 1., image_input, "val")

            # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
