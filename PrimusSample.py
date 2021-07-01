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

from Parameters import Parameters as PARAM
import ImageProcessing

import os
import PIL

import cv2
import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt

class PrimusSample:

    DATASET_DIR = './Complete_Primus'
    TEMP_IMG_FILENAME = 'TMP_IMAGE.png'
    
    def __init__(self, id : str):
        self.id 					=	id
        base_filename               =   F'{self.DATASET_DIR}/{self.id}/{self.id}'
        self.audio_wav_path 		=	F'{base_filename}.wav'
        self.score_semantic         =	F'{base_filename}.semantic'
        self.score_midi             =   F'{base_filename}.mid'
        if os.path.isfile(F'{base_filename}_spec.png'):
            self.audio_img_path     =   F'{base_filename}_spec.png'
        else:
            self.audio_img_path     =   'IMG_NOT_CREATED'

    def get_spectrogram_db(self):
        audio_time_series, achieved_sample_rate = librosa.load(self.audio_wav_path, mono=True)
        X = librosa.stft(audio_time_series, hop_length=PARAM['STFT']['HOP_LENGTH'], win_length=PARAM['STFT']['WIN_LENGTH'])
        D = librosa.amplitude_to_db(np.abs(X), ref=np.max)
        return D

    def get_semantic_tokens(self):
        tokens = open(self.score_semantic, 'r').readline().split('\t')
        tokens.pop()
        return tokens

    def save_spectogram_into_dataset(self):
        plt.clf()
        plt.cla()
        plt.close()
        plt.axis('off')
        D = self.get_spectrogram_db()
        img = librosa.display.specshow(D, x_axis='linear',
                                	   hop_length=PARAM['STFT']['HOP_LENGTH'], y_axis=PARAM['SPEC']['Y_AXIS_SCALE'])
        img.figure.savefig(self.TEMP_IMG_FILENAME, bbox_inches='tight', pad_inches=0)
        pil_image = PIL.Image.open(self.TEMP_IMG_FILENAME)
        pil_image = pil_image.resize((D.shape[1] * PARAM['SPEC']['IMG_WIDTH_PER_FRAME'], PARAM['SPEC']['IMG_HEIGHT']), resample=PIL.Image.LANCZOS)
        filename = self.audio_wav_path.replace('.wav', '_spec.png')
        pil_image.save(filename)
        self.audio_img_path = filename
        os.system(F'rm {self.TEMP_IMG_FILENAME}')

    def read_img(self):
        if self.audio_img_path != 'IMG_NOT_CREATED':
            img = cv2.imread(self.audio_img_path)
            if img is None:
                raise Exception(F'Could not read image {self.audio_img_path}')
            return img
        else:
            raise Exception(F'Cannot read spectrogram from {self.id} because it doesn\'t exist')
    
    def get_preprocesssed_img(self):
        return ImageProcessing.preprocess_img(self.read_img())