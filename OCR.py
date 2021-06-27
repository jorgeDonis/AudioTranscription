
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

from tensorflow.keras import callbacks
from tensorflow.python.keras.callbacks import ModelCheckpoint
from OCR_Dataset import OCR_Dataset

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow.keras as K

import cv2
import glob
import re
from os import system
from Levenshtein import distance as levenshtein_distance

NUM_IMAGES_DATASET = 384151
BATCH_SIZE = 32
IMG_HEIGHT = 32
CHAR_LIST = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
BLANK_CHARACTER = len(CHAR_LIST)
POOLING_RATIO = 8
EPOCHS = 1
MODEL_NAME = "ocr_model.h5"

def encode_str(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        dig_lst.append(CHAR_LIST.index(char))
    return dig_lst

def load_image_label(image_path):
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

def pad_img_vertical(img, height):
    return np.pad(img, ( (0, height - img.shape[0]), (0, 0), (0, 0) ), 'constant', constant_values=( (0, 0), (0, 0), (0, 0) ))

def pad_img_horizontal(img, max_img_len):
    return np.pad(img, ( (0, 0), (0, max_img_len - img.shape[1]), (0, 0) ), 'constant', constant_values= ( (0, 0), (0, 0), (0, 0) ))
    
def pad_label(label, max_label_len):
    for i in range(0, max_label_len - len(label)):
        label.append(BLANK_CHARACTER)
    return label

def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def num_repeated_chars(label):
    num_repeated_chars = 0
    if len(label) == 0:
        return num_repeated_chars
    previous_char = label[0]
    for i in range(1, len(label)):
        if label[i] == previous_char:
            num_repeated_chars += 1
        previous_char = label[i]
    return num_repeated_chars

def images_wide_enough(imgs, labels):
    for img, label in zip(imgs, labels):
        #CHECK FOR BLANK CHARACTER (more timesteps required)
        if img.shape[1] // POOLING_RATIO < (len(label) + num_repeated_chars(label)):
            return False
    return True

# Yields training data in batches
def train_generator():
    dataset = OCR_Dataset()
    i = 0
    images = []
    labels = []
    for image_path in glob.iglob(dataset.DEBUG_DATASET_DIR_TRAIN + "*.jpg"):
        img, label = load_image_label(image_path)
        if (img is None):
            continue
        images.append(img)
        labels.append(label)
        i += 1
        if (i == BATCH_SIZE):
            max_image_len = max([x.shape[1] for x in images])
            max_label_len = max([len(x) for x in labels])
            if images_wide_enough(images, labels):
                original_image_lengths_after_pooling = np.array([x.shape[1] // POOLING_RATIO for x in images])
                original_label_lens = np.array([len(x) for x in labels])
                images = [pad_img_horizontal(x, max_image_len) for x in images]
                labels = [encode_str(x) for x in labels]
                labels = [pad_label(x, max_label_len) for x in labels]
                images = np.array(images)
                labels = np.array(labels)
                inputs_fit = {
                    'padded_images': images,
                    'padded_labels': labels,
                    'original_image_lengths_after_pooling': original_image_lengths_after_pooling,
                    'original_label_lengths': original_label_lens
                }
                outputs_fit = { 'ctc': np.zeros([BATCH_SIZE]) } 
                yield inputs_fit, outputs_fit
            images = []
            labels = []
            i = 0


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_activation_training_models():
    inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, None, 1), name='padded_images')

    conv_1 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
    conv_2 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
    conv_3 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
    batch_norm = tf.keras.layers.BatchNormalization()(conv_3)
    pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(batch_norm)
    conv_4 = tf.keras.layers.Conv2D(512, (2,2), activation = 'relu', padding='same')(pool_3)
    permute = tf.keras.layers.Permute((2, 1, 3))(conv_4)
    reshape = tf.keras.layers.Reshape((-1, 512 * IMG_HEIGHT // POOLING_RATIO))(permute)
    blstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 128, input_shape=[None], return_sequences=True, dropout = 0.2))(reshape)
    blstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 128, input_shape=[None], return_sequences=True, dropout = 0.2))(blstm_1)
    outputs = tf.keras.layers.Dense(len(CHAR_LIST) + 1, activation = 'softmax')(blstm_2)
    act_model = tf.keras.Model(inputs, outputs)

    labels = tf.keras.layers.Input(dtype='float32', shape=[None], name='padded_labels')
    input_length = tf.keras.layers.Input(dtype='int64', shape=[1], name='original_image_lengths_after_pooling')
    label_length = tf.keras.layers.Input(dtype='int64', shape=[1], name='original_label_lengths')

    loss_out = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1), name='ctc')([outputs, labels, input_length, label_length])
    training_model = tf.keras.Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    training_model.compile(loss = { 'ctc' : lambda y_true, y_pred: y_pred }, optimizer = 'adam')

    return training_model, act_model

generator = train_generator()
training_model, act_model = get_activation_training_models()

def get_loss(model, val_data):
    return 0

def train_model(model, input_generator, val_data):
    for i in range(0, EPOCHS):
        print(F"Training EPOCH {i}")
        model.fit(x = generator)
        print(F"Validation loss: {get_loss(model, val_data)}")
    model.save(MODEL_NAME)


def char_pred_list_to_string(char_list):
    previous_char = char_list[0]
    char_list_no_repeated = [previous_char]
    for i in range(1, len(char_list)):
        if char_list[i] != previous_char:
            char_list_no_repeated.append(char_list[i])
        previous_char = char_list[i]
    char_list_no_repeated = [char for char in char_list_no_repeated if char != ' ']
    return char_list_no_repeated


def print_image_text(image_path):
    img, label = load_image_label(image_path)
    if (img is None):
        print(F"Image {image_path} could not be loaded")
        return
    input = { 'padded_images' : np.array([img]) }
    prediction = act_model.predict(input)
    best_indices = [ np.argmax(x) for x in prediction[0]]
    char_pred_list = [ CHAR_LIST[x] if x != BLANK_CHARACTER else ' ' for x in best_indices]
    print(F"Original: {label}, predicted: {char_pred_list_to_string(char_pred_list)}")

# training_model.fit(x=generator)

# print_image_text("./OCR_dataset_3/23_Wasp_85572.jpg")
# print_image_text("./OCR_dataset_3/21_AFFECTED_1392.jpg")
# print_image_text("./OCR_dataset_3/138_broodily_9767.jpg")
# print_image_text("./OCR_dataset_3/210_canoes_11214.jpg")
# print_image_text("./OCR_dataset_3/194_vicksburg_84553.jpg")

