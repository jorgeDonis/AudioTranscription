# Copyright (C) 2020 JORGE DONIS DEL ÁLAMO

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from typing import List, Tuple
import numpy as np
import tensorflow as tf
import cv2
import glob
import re
import os

BATCH_SIZE = 4
IMG_HEIGHT = 32
CHAR_LIST = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
BLANK_CHARACTER = len(CHAR_LIST)

def encode_str(txt : str) -> List[int]:
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        dig_lst.append(CHAR_LIST.index(char))
    return dig_lst

def load_image_label(image_path : str) -> Tuple[np.ndarray, str]:
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY) 
    img = np.expand_dims(img, axis=2)
    img = pad_img_vertical(img, IMG_HEIGHT)
    img = img / 255
    regex = r".*\d+_([a-zA-Z0-9]+)_\d+\..*"
    label = re.findall(regex, image_path)[0]
    return img, label

def pad_img_vertical(img : np.ndarray, height : int) -> np.ndarray:
    return np.pad(img, ( (0, height - img.shape[0]), (0, 0), (0, 0) ), 'constant', constant_values=( (0, 0), (0, 0), (0, 0) ))

def pad_img_horizontal(img, max_img_len) -> np.ndarray:
    return np.pad(img, ( (0, 0), (0, max_img_len - img.shape[1]), (0, 0) ), 'constant', constant_values= ( (0, 0), (0, 0), (0, 0) ))
    
def pad_label(label : List[int], max_label_len : int) -> List[int]:
    for i in range(0, max_label_len - len(label)):
        label.append(BLANK_CHARACTER)
    return label

def show_img(img) -> None:
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Yields training data in batches
def train_generator() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i = 0
    images = []
    labels = []
    for image_path in glob.iglob("./OCR_dataset_2/*.jpg"):
        img, label = load_image_label(image_path)
        images.append(img)
        labels.append(label)
        i += 1
        if (i == BATCH_SIZE):
            max_image_len = max([x.shape[1] for x in images])
            max_label_len = max([len(x) for x in labels])
            original_img_lens = np.array([x.shape[1] for x in images])
            original_label_lens = np.array([len(x) for x in labels])
            images = [pad_img_horizontal(x, max_image_len) for x in images]
            labels = [encode_str(x) for x in labels]
            labels = [pad_label(x, max_label_len) for x in labels]
            images = np.array(images)
            labels = np.array(labels)
            yield images, labels, original_img_lens, original_label_lens
            images = []
            labels = []
            i = 0

generator = train_generator()
for i in range(0, 3):
    images, labels, original_img_lens, original_label_lens = next(generator)
    for image in images:
        show_img(image)


