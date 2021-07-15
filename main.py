
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

import sys
from PrimusSample import PrimusSample
import tensorflow as tf
import PrimusDataset
import cv2
import matplotlib.pyplot as plt
import Model

# PrimusDataset.gen_train_test_ids(75000, 5000)

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

activation_model, training_model = Model.get_activation_training_models()
activation_model.summary()
training_generator = PrimusDataset.train_generator()
validation_generator_factory = PrimusDataset.val_generator_factory

Model.train_model(training_model, activation_model, training_generator, validation_generator_factory)

Model.test_all_images(activation_model)

# model = tf.keras.models.load_model('cnn_muy_wena.h5')