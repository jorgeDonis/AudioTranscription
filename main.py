
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

import os
import sys

import librosa
from numpy.core.fromnumeric import shape
from PrimusSample import PrimusSample
import tensorflow as tf
import PrimusDataset
import cv2
import matplotlib.pyplot as plt
import Model
import numpy as np
import ImageProcessing
import math

# PrimusDataset.gen_train_test_ids(75000, 5000)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# activation_model, training_model = Model.get_activation_training_models()
# training_generator = PrimusDataset.train_generator()
validation_generator_factory = PrimusDataset.val_generator_factory

# Model.train_model(training_model, activation_model, training_generator, validation_generator_factory)

activation_model = tf.keras.models.load_model('cnn_mel_0.41.h5')
print(Model._get_loss(activation_model, validation_generator_factory()))
# Model.test_all_images(activation_model)

# train_histo = [ line.replace('\n', '').replace(' ', '').split(',') for line in open('fit_histo_modelo_weno_replica.csv', 'r').readlines() ]
# train_histo = [ [ float(epoch[1]), float(epoch[2]) ] for epoch in train_histo ] 
# x = np.arange(1, 15, 1)
# in_loss = np.array([ epoch[0] for epoch in train_histo ])
# out_loss = np.array([ epoch[1] for epoch in train_histo ])

# fig, ax1 = plt.subplots()
# ax1.plot(x, in_loss, 'b')
# ax1.set_xlabel('Epoch')
# # Make the y-axis label, ticks and tick labels match the line color.
# ax1.set_ylabel('CTC Loss', color='b')
# ax1.tick_params('y', colors='b')

# ax2 = ax1.twinx()
# ax2.plot(x, out_loss, 'r')
# ax2.set_ylabel('Edit distance', color='r')
# ax2.tick_params('y', colors='r')

# fig.tight_layout()
# plt.show()

# fmin = 27.5
# fmax = 3520
# n_fft = 1024
# window_length = 1024
# hop_length = 128
# img_height = 192

# while True:
#     plt.clf()
#     plt.cla()
#     plt.close()
#     plt.axis('off')

#     sample = PrimusDataset.get_random_sample()
#     y, sr = librosa.load(sample.audio_wav_path, mono=True)
#     # S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=window_length)
#     S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=window_length)
#     S_db = librosa.amplitude_to_db(S)
#     vmin = np.amin(S_db)
#     vmax = np.amax(S_db)
#     print(F'Minimmum value: {vmin}, Maximmum value: {vmax}')

#     img = librosa.display.specshow(S_db, fmin=fmin, fmax=fmax, x_axis='time', y_axis='mel',
#                                         hop_length=hop_length, vmin=-70, vmax=8)

#     img.figure.savefig('temp_img.png', bbox_inches='tight', pad_inches=0)
#     img = cv2.imread('temp_img.png')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#     downscale_ratio = img_height / img.shape[0]
#     img = cv2.resize(img, (math.ceil(img.shape[1] * downscale_ratio), img_height), interpolation=cv2.INTER_LANCZOS4)
#     img = np.expand_dims(img, axis=2)
#     img = img / 255

#     first_black_col = img.shape[1] - 1
#     for i in range(0, img.shape[1]):
#         column_is_black = True
#         for j in range(0, img_height):
#             if img[j][img.shape[1] - i - 1] != 0:
#                 column_is_black = False
#                 break
#         if column_is_black:
#             first_black_col = img.shape[1] - i - 1
#         else:
#             break

#     img = img[:,0:first_black_col]

#     plt.clf()
#     plt.cla()
#     plt.close()
#     plt.axis('off')
#     print(f'id = {sample.id}')
#     if 'rest' in sample.get_semantic_tokens()[len(sample.get_semantic_tokens()) - 2]:
#         ImageProcessing.show_img(cv2.imread(sample.score_img))
#         os.system(f'cvlc {sample.audio_wav_path}')