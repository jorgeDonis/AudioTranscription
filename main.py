
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

import tensorflow as tf
import PrimusDataset
import cv2
import matplotlib.pyplot as plt
import Model

# def show_img(img):
#     plt.imshow(img)
#     plt.show()

# for inputs_fit, outputs_fit in PrimusDataset.train_generator():
#     imgs = inputs_fit['padded_images']
#     encodings = inputs_fit['padded_encodings']
#     img_lens = inputs_fit['original_image_lengths_after_pooling']
#     encodings_lens = inputs_fit['original_encoding_lengths']
#     for img, encoding, img_len, encoding_len in zip(imgs, encodings, img_lens, encodings_lens):
#         if img_len < encoding_len:
#             print("ENCODING")
#             print(encoding)
#             print(F"Img len: {img_len}")
#             print(F"Encoding len: {encoding_len}")
#             show_img(img)

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# activation_model, training_model = Model.get_activation_training_models()
# training_generator = PrimusDataset.train_generator()
# validation_generator = PrimusDataset.validation_generator()

# Model.train_model(training_model, activation_model, training_generator, validation_generator)

model = tf.keras.models.load_model('cnn.h5')
Model.test_all_images(model)
