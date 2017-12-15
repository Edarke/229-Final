import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sklearn.model_selection as sk
import labels


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def early_stopping(val_loss_history, patience):
    """
    Returns true if the validation loss has increased at least `patience` times in a row.
    :param val_loss_history:
    :param patience: Number of times that val_loss is allowed to increase consecutively.
    :return:
    """
    for i in range(patience + 1):
        if val_loss_history[-(i + 2)] > val_loss_history[-1]:
            return False
    return True


def crop_center(img, startx, starty, width, height):
    return img[starty:starty + height, startx:startx + width]


def compute_label_frequency(image_paths, label_paths):
    categories = np.array([*set(labels.id_to_trainId.values())], dtype=np.int)
    counts = np.zeros(categories.shape[0] + 1)

    print(categories)
    for image_file in image_paths:
        gt_image = scipy.misc.imread(label_paths[image_file])
        gt_image = labels.id_to_trainId_map_func(gt_image)
        gt_image = gt_image.flatten()
        diff = np.setdiff1d(categories, gt_image)
        for label in diff:
            counts[label] += 1.

    total_examples = float(len(image_paths))
    for i in range(counts.shape[0]):
        counts[i] = 1 - counts[i] / total_examples
    print(counts)


def gen_batch_function(data_folder, image_shape, should_crop):
    """gt_image_file
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    print("Cropping images", should_crop)

    def get_batches_fn(batch_size, get_train=True, use_extra=False):
        subfolder = "training" if get_train else "trainings"
        """
        Create batches of training data
        :param get_train:
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        max_width, max_height = image_shape

        image_paths = []
        label_paths = {}
        # For train subfolders, first add the extras
        # if (use_extra and subfolder == "train"):
        #     image_paths = glob(os.path.join("data", "leftImg8bit", subfolder + "_extra", '**/*bit.png'))
        #     label_paths = {
        #         re.sub("gtCoarse/", "leftImg8bit/", re.sub('_gtCoarse_labelIds', '_leftImg8bit', path)): path
        #         for path in glob(os.path.join("data", 'gtCoarse', subfolder + "_extra", '**/*Ids.png'))}
        #
        image_paths += glob(os.path.join("data", "data_road", subfolder, 'gt_image_2', '*_road_*.png'))
        label_paths.update({
            re.sub('gt_image_2', 'image_2', re.sub('_road_', '_', path)): path
            for path in image_paths})
        image_paths = list(label_paths.keys())

        print("Total images: " + str(len(image_paths)) + " Total labels: " + str(len(label_paths)))

        # compute_label_frequency(image_paths, label_paths)

        for batch_i in range(0, int(len(image_paths)), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[image_file]

                image = scipy.misc.imread(image_file)
                gt_image = (scipy.misc.imread(gt_image_file))
                # if should_crop:
                #     orig_height, orig_width, _ = image.shape
                #     xoffset = np.random.randint(0, orig_width - max_width)
                #     yoffset = np.random.randint(0, orig_height - max_height)
                #     image = crop_center(image, xoffset, yoffset, max_width, max_height)
                #     gt_image = crop_center(gt_image, xoffset, yoffset, max_width, max_height)
                # elif False:
                image = scipy.misc.imresize(image, (375, 1242), interp="nearest")
                gt_image = scipy.misc.imresize(gt_image, (375, 1242), interp="nearest")
                gt_image = (gt_image[:, :, 2] != 0).astype(int)

                # gt_image = labels.id_to_trainId_map_func(gt_image)
                images.append(image)
                gt_images.append(gt_image)
            yield np.array(images), np.array(gt_images)

    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, image_shape, data_set):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    if data_set == "test":
        path = "data/leftImg8bit/test/**/*.png"
    elif data_set == "train":
        path = "data/leftImg8bit/train/**/*.png"
    elif data_set == "val":
        path = "data/leftImg8bit/val/**/*.png"
    else:
        raise Exception("Folder not recognized")

    for image_file in glob(os.path.join(path)):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        if keep_prob == None:
           image = image.reshape(-1, 3)
           im_softmax = sess.run ([tf.nn.softmax(logits)],
                                  {image_pl: image})[0]

           image = image.reshape (image_shape[0], image_shape[1],  3)
           mask = np.argmax (im_softmax, axis = 1)

        else:
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image]})[0]
            mask = np.argmax(im_softmax, axis=1)

        mask = mask.reshape([image_shape[0], image_shape[1]])

        mask2 = labels.get_color_matrix(mask)
        mask = scipy.misc.toimage(mask2, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, data_set):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()) + "/" + data_set)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, image_shape, data_set)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
