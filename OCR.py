
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

from OCR_Dataset import OCR_Dataset

import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np

import tensorflow as tf
import tensorflow.keras as K

import cv2
import glob
import re
from os import system

NUM_IMAGES_DATASET = 384151
BATCH_SIZE = 4
IMG_HEIGHT = 32
CHAR_LIST = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
BLANK_CHARACTER = len(CHAR_LIST)
POOLING_RATIO = 8

def encode_str(txt : str) -> List[int]:
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        dig_lst.append(CHAR_LIST.index(char))
    return dig_lst

def load_image_label(image_path : str) -> Tuple[np.ndarray, str]:
    img = cv2.imread(image_path)
    if (img is None):
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
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
    plt.imshow(img, cmap='gray')
    plt.show()

def images_wide_enough(img_lens, label_lens):
    for i in range(0, len(img_lens)):
        if img_lens[i] < label_lens[i]:
            return False
    return True

# Yields training data in batches
def train_generator() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = OCR_Dataset()
    i = 0
    images = []
    labels = []
    for image_path in glob.iglob(dataset.DATASET_DIR + "*.jpg"):
        img, label = load_image_label(image_path)
        if (img is None):
            continue
        images.append(img)
        labels.append(label)
        i += 1
        if (i == BATCH_SIZE):
            max_image_len = max([x.shape[1] for x in images])
            max_label_len = max([len(x) for x in labels])
            #check that all images in the batch are wide enough
            if ((max_image_len // POOLING_RATIO) >= max_label_len):
                original_img_lens = np.array([x.shape[1] // POOLING_RATIO for x in images])
                original_label_lens = np.array([len(x) for x in labels])
                images = [pad_img_horizontal(x, max_image_len) for x in images]
                labels = [encode_str(x) for x in labels]
                labels = [pad_label(x, max_label_len) for x in labels]
                images = np.array(images)
                labels = np.array(labels)
                inputs_fit = {
                    'padded_images': images,
                    'padded_labels': labels,
                    'original_image_lengths': original_img_lens,
                    'original_label_lengths': original_label_lens
                }
                outputs_train = { 'ctc': np.zeros([BATCH_SIZE]) } 
                yield inputs_fit, outputs_train, [None]
            images = []
            labels = []
            i = 0

def print_debug_batch_loss(args):
    y_pred, labels, input_length, label_length = args
    print(F"[INFO] y_pred = {y_pred}")
    print(F"[INFO] labels = {labels}")
    print(F"[INFO] input_length = {input_length}")
    print(F"[INFO] label_length = {label_length}")
    print("\n")


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    print_debug_batch_loss(args)
    return K.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_activation_training_models() -> Tuple[object, object]:
    inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, None, 1), name='padded_images')

    conv_1 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
    conv_2 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
    conv_3 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
    batch_norm = tf.keras.layers.BatchNormalization()(conv_3)
    pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(batch_norm)
    conv_4 = tf.keras.layers.Conv2D(512, (2,2), activation = 'relu', padding='same')(pool_3)
    reshape = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [tf.shape(conv_4)[0], -1, 512 * 4])) (conv_4)
    blstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 128, input_shape=[None], return_sequences=True, dropout = 0.2))(reshape)
    blstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 128, input_shape=[None], return_sequences=True, dropout = 0.2))(blstm_1)
    outputs = tf.keras.layers.Dense(len(CHAR_LIST) + 1, activation = 'softmax')(blstm_2)
    act_model = tf.keras.Model(inputs, outputs)

    labels = tf.keras.layers.Input(dtype='int64', shape=[None], name='padded_labels')
    input_length = tf.keras.layers.Input(dtype='int64', shape=[1], name='original_image_lengths')
    label_length = tf.keras.layers.Input(dtype='int64', shape=[1], name='original_label_lengths')

    loss_out = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
    training_model = tf.keras.Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    training_model.compile(loss = { 'ctc' : lambda y_true, y_pred: y_pred }, optimizer = 'adam')

    return training_model, act_model

generator = train_generator()
training_model, act_model = get_activation_training_models()
filepath="best_model.hdf5"
checkpoint = K.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
# act_model.summary()
training_model.summary()

num_bad_images = 0
for i in range(0, 384151 // BATCH_SIZE):
    # print(F"i = {i}")
    inputs_fit, outputs_train, shape = next(generator)
    for j in range(0, BATCH_SIZE):
        if ((inputs_fit['padded_images'][j].shape[1] // POOLING_RATIO) < inputs_fit['original_label_lengths'][j]):
            img = inputs_fit['padded_images'][j]
            show_img(img)
            print(F"Padded_width = {img.shape[1]}")
            print(F"Original_width = {inputs_fit['original_image_lengths'][j]}")
            print(F"Original label length = {inputs_fit['original_label_lengths'][j]}")
            num_bad_images = num_bad_images + 1
        
print(F"Number of bad images : {num_bad_images} ({num_bad_images / 384151 * 100}%)")

# training_model.fit(x = generator, callbacks = callbacks_list)
