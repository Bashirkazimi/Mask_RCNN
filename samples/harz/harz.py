"""
Mask R-CNN
Configurations and data loading code for the harz dataset.
This is a duplicate of the code in the noteobook train_harz.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import warnings
warnings.filterwarnings("ignore")
import os
import sys
import math
import random
import numpy as np
import cv2
import imgaug

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_DATA_DIR="/notebooks/tmp/data/DTM_DATA"


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import utils
import skimage.io
from scipy import ndimage
from sklearn.model_selection import train_test_split
from osgeo import gdal, ogr, osr
import gdalconst


class HarzConfig(Config):
    """Configuration for training on harz dataset.
    Derives from the base Config class and overrides values specific
    harz dataset.
    """
    # Give the configuration a recognizable name
    NAME = "harz"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 12

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = (4886 // (GPU_COUNT*IMAGES_PER_GPU)) + 1

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = (611 // (GPU_COUNT*IMAGES_PER_GPU)) + 1


class HarzDataset(utils.Dataset):
    """Generates the harz dataset. The dataset consists of bomb craters, 
    meiler (charcoal kilns), barrows, and pinge (mining sinkholes)
    The images are generated on the fly. No file access required.
    """

    def get_file_names(self, root, subset='train'):
        """
        returns filenames for images in root and for the given subset (train,test,valid!)
        """
        with open(os.path.join(root, 'labels.txt'), 'r') as f:
            lines = f.readlines()
            labels = np.empty((len(lines),))
            file_names = []
            for e, filename in enumerate(lines):
                file_name, classlabel = filename.split()
                labels[e] = int(classlabel)
                file_names.append(file_name)

        train_file_names, test_file_names, train_labels, test_labels = \
            train_test_split(
                file_names,
                labels,
                test_size=0.1,
                stratify=labels,
                random_state=42
            )

        train_file_names, validation_file_names, train_labels, valid_labels = \
            train_test_split(
                train_file_names,
                train_labels,
                test_size=len(test_file_names),
                stratify=train_labels,
                random_state=42
            )

        if subset == 'train':
            return train_file_names
        elif subset == 'test':
            return test_file_names
        else:
            return validation_file_names

    def load_data(self, dataset_dir, subset, height=256, width=256):
        """Generate the requested number of synthetic images.
        data directory for images.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("harz", 1, "bombs")
        self.add_class("harz", 2, "meiler")
        self.add_class("harz", 3, "barrows")
        self.add_class("harz", 4, "pinge")
        self.split = subset
        self.filenames = self.get_file_names(dataset_dir,subset)
        # self.imgs = [os.path.join(dataset_dir, 'RGB', f) for f in self.filenames]
        # self.masks = [os.path.join(dataset_dir, 'y', f) for f in self.filenames]

        # Add images

        for f in self.filenames:
            self.add_image("harz", 
                image_id=f, 
                path=os.path.join(dataset_dir, 'RGB', f), 
                width=width, 
                height=height,
                mask_path=os.path.join(dataset_dir, 'y', f)
                )

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image = skimage.io.imread(info['path'], plugin='pil')
        return image

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "harz":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_mask_helper(self, mask_path):
        "returns instance mask for the label file at mask_path"
        mask = skimage.io.imread(mask_path, plugin='pil')
        obj_ids = np.unique(mask)
        # exclude background
        obj_ids = obj_ids[1:]

        labeled_array, num_features = ndimage.label(mask)
        masks = np.zeros((mask.shape[0], mask.shape[1], num_features))
        labels = []
        for i in range(1, num_features+1):
            pos = np.where(labeled_array == i)
            masks[:,:,i-1][pos] = 1
            labels.append(mask[pos][0])

        return masks.astype(np.bool), np.array(labels, dtype=np.int32)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask_path = info['mask_path']
        return self.load_mask_helper(mask_path)


############################################################
# Evaluate and print mAP
############################################################

def evaluate(model, dataset_val, inference_config):
    APs = []
    for image_id in dataset_val.image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        
    print("mAP: ", np.mean(APs))


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Harz Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on Harz Dataset")
    parser.add_argument('--dataset', required=False,
                        default=DEFAULT_DATA_DIR,
                        metavar="/path/to/data/dir/",
                        help='Directory of the Harz dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or one of 'imagenet', 'last', or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--limit', required=False,
    #                     default=500,
    #                     metavar="<image count>",
    #                     help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = HarzConfig()
    else:
        class InferenceConfig(HarzConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    if args.model.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = HarzDataset()
        dataset_train.load_data(args.dataset, 'train')
        dataset_train.prepare()

        # Validation dataset
        dataset_val = HarzDataset()
        dataset_val.load_data(args.dataset, 'validation')
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=20,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=100,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = HarzDataset()
        dataset_val.load_data(args.dataset, 'test')
        dataset_val.prepare()
        print("Running harz evaluation on test data")
        evaluate(model, dataset_val, config)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))

