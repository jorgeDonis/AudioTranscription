# Copyright (C) 2021 JORGE DONIS DEL ÁLAMO

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

import os
import sys
from PrimusSample import PrimusSample
from typing import List
from Parameters import Parameters as PARAM
import PrimusDataset

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as Layer

from tabulate import tabulate
import edit_distance

import numpy as np

semantic_translator = PrimusDataset.get_semantic_translator()

def _ctc_lambda_func(args):
    y_pred, encodings, input_length, encoding_length = args
    return K.backend.ctc_batch_cost(encodings, y_pred, input_length, encoding_length)

def get_activation_training_models():
    inputs = Layer.Input(shape=(PARAM['SPEC']['IMG_HEIGHT'], None, 1), name='padded_images')

    conv_1_1 = Layer.Conv2D(64, (3,3), activation = 'gelu', padding='same')(inputs)
    conv_1_2 = Layer.Dropout(0.2)(conv_1_1)

    pool_1 = Layer.MaxPool2D(pool_size=(2, 2), strides=2)(conv_1_2)
    conv_2 = Layer.Conv2D(64, (3,3), activation = 'gelu', padding='same')(pool_1)
    batch_norm_1 = Layer.BatchNormalization()(conv_2)
    pool_3 = Layer.MaxPool2D(pool_size=(2, 2))(batch_norm_1)
    conv_4 = Layer.Conv2D(64, (3,3), activation = 'gelu', padding='same')(pool_3)
    pool_4 = Layer.MaxPool2D(pool_size=(2, 2))(conv_4)
    conv_5 = Layer.Conv2D(128, (3,3), activation='gelu', padding='same')(pool_4)
    dropout = Layer.Dropout(0.2)(conv_5)
    permute = Layer.Permute((2, 1, 3))(dropout)
    reshape = Layer.Reshape((-1, 128 * PARAM['SPEC']['IMG_HEIGHT'] // PARAM['TRAINING']['POOLING_RATIO']))(permute)
    blstm_1 = Layer.Bidirectional(Layer.LSTM(units = 128, input_shape=[None], return_sequences=True, dropout=0.2))(reshape)
    batch_norm_2 = Layer.BatchNormalization()(blstm_1)
    blstm_2 = Layer.Bidirectional(Layer.LSTM(units = 128, input_shape=[None], return_sequences=True, dropout=0.2))(batch_norm_2)
    outputs = Layer.Dense(semantic_translator.blank_class + 1, activation = 'softmax')(blstm_2)
    act_model = tf.keras.Model(inputs, outputs)

    encodings = Layer.Input(dtype='int32', shape=[None], name='padded_encodings')
    input_length = Layer.Input(dtype='int32', shape=[1], name='original_image_lengths_after_pooling')
    encoding_length = Layer.Input(dtype='int32', shape=[1], name='original_encoding_lengths')

    loss_out = Layer.Lambda(_ctc_lambda_func, output_shape=(1), name='ctc')([outputs, encodings, input_length, encoding_length])
    training_model = tf.keras.Model(inputs=[inputs, encodings, input_length, encoding_length], outputs=loss_out)
    training_model.compile(loss = { 'ctc' : lambda y_true, y_pred: y_pred }, optimizer = 'adamax')

    return act_model, training_model


def _remove_repeated_tokens(sequence):
    previous_token = sequence[0]
    sequence_no_repeated = [previous_token]
    for i in range(1, len(sequence)):
        if sequence[i] != previous_token:
            sequence_no_repeated.append(sequence[i])
        previous_token = sequence[i]
    return sequence_no_repeated

def _decode_softmax_tokens(prediction) -> List[str]:
    best_indices = [ np.argmax(x) for x in prediction ]
    indices_no_repeated = _remove_repeated_tokens(best_indices)
    token_pred_list = [ semantic_translator.decode_semantic_class_index(x) for x in indices_no_repeated if x != semantic_translator.blank_class]
    return token_pred_list

def _decode_softmax_indices(prediction) -> List[int]:
    best_indices = [ np.argmax(x) for x in prediction ]
    indices_no_repeated = _remove_repeated_tokens(best_indices)
    class_pred_list = [ x for x in indices_no_repeated if x != semantic_translator.blank_class]
    return class_pred_list

def _edit_distance(a: List, b: List) -> int:
    sm = edit_distance.SequenceMatcher(a, b)
    return sm.distance()

def _get_loss(model, val_generator):
    predictions = []
    true_encodings = []
    for batch_inputs, batch_encodings in val_generator:
        batch_predictions = model.predict(batch_inputs)
        for prediction, encoding in zip(batch_predictions, batch_encodings):
            predictions.append(_decode_softmax_indices(prediction))
            true_encodings.append(encoding)
    total_loss = 0
    for i in range(0, len(predictions)):
        l_d = _edit_distance(true_encodings[i], predictions[i])
        total_loss += l_d / len(true_encodings[i])
    return total_loss / len(predictions)

def train_model(train_model, act_model, input_generator, val_generator_factory, 
                saved_model_filename="cnn.h5", training_history_filename="fit_histo.csv"):
    lowest_val_loss = float('inf')
    histo_file = open(training_history_filename, 'a')
    for i in range(0, PARAM['TRAINING']['EPOCHS']):
        val_generator = val_generator_factory()
        print(F"Training EPOCH {i + 1}")
        history = train_model.fit(x=input_generator, steps_per_epoch=PrimusDataset.num_train_samples() / PARAM['TRAINING']['BATCH_SIZE'], epochs=1)
        batch_in_loss = history.history['loss'][0]
        batch_val_loss = _get_loss(act_model, val_generator)
        histo_file.write(F"{i}, {batch_in_loss}, {batch_val_loss}\n")
        print(F"Validation loss: {batch_val_loss}")
        if batch_val_loss < lowest_val_loss:
            act_model.save(saved_model_filename)

def _print_predicted_vs_true(predicted, true):
    max_len = max(len(predicted), len(true))
    if max_len == len(predicted):
        true.append([ ' ' for i in range(0, max_len - len(true)) ])
    else:
        predicted.append([ ' ' for i in range(0, max_len - len(predicted)) ])
    table = [ [true_token, predicted_token] for predicted_token, true_token in zip(predicted, true) ]
    print(tabulate(table, headers=['TRUE', 'PREDICTED']))

def test_all_images(model):
    train_ids, test_ids = PrimusDataset.get_train_test_ids()
    for id in test_ids:
        sample = PrimusSample(id)
        img = sample.get_preprocesssed_img()
        input = { 'padded_images' : np.array([img]) }
        prediction = _decode_softmax_tokens(model.predict(input)[0])
        _print_predicted_vs_true(prediction, sample.get_semantic_tokens())
        sys.stdin.read(1)
