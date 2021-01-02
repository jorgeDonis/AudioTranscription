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


#This class is used to access the MTD dataset.

import os
import glob
import json
import PIL
import subprocess

import IPython.display as ipd
import numpy as np
import pandas as pd
from pretty_midi import pretty_midi
import librosa
import librosa.display
from matplotlib import pyplot as plt
from matplotlib import patches

import matplotlib.pyplot as plt

class MTDSample:

  ##todo save: image (log and no log), frequency matrix

  BASE_DATASET_PATH = './MTD'
  SPECTROGRAM_DISPLAY_HOP_LENGTH = 512
  WIN_LENGTH = 2048
  TEMP_IMG_FILENAME = "TMP_IMAGE.png"
  IMG_WIDTH = 334
  IMG_HEIGHT = 217

  def __init__(self, id):
    self.id = id
    self.corr_mid = self.get_file(os.path.join(MTDSample.BASE_DATASET_PATH, 'data_EDM-corr_MID', f'MTD{self.id}_*.mid'))
    self.corr_csv = self.get_file(os.path.join(MTDSample.BASE_DATASET_PATH, 'data_EDM-corr_CSV', f'MTD{self.id}_*.csv'))
    self.alig_mid = self.get_file(os.path.join(MTDSample.BASE_DATASET_PATH, 'data_EDM-alig_MID', f'MTD{self.id}_*.mid'))
    self.alig_csv = self.get_file(os.path.join(MTDSample.BASE_DATASET_PATH, 'data_EDM-alig_CSV', f'MTD{self.id}_*.csv'))
    self.score_pdf = self.get_file(os.path.join(MTDSample.BASE_DATASET_PATH, 'data_SCORE_IMG', f'MTD{self.id}_*.pdf'))
    self.score_xml = self.get_file(os.path.join(MTDSample.BASE_DATASET_PATH, 'data_SCORE_XML', f'MTD{self.id}_*.xml'))
    self.score_humdrum = self.get_file(os.path.join(MTDSample.BASE_DATASET_PATH, 'data_SCORE_HUMDRUM', f'MTD{self.id}_*.txt'))
    try:
      self.spectogram_img = self.get_file(os.path.join(MTDSample.BASE_DATASET_PATH, 'data_AUDIO_IMG', f'MTD{self.id}_*.png'))
    except:
      self.spectogram_img = 'None'    
    self.json = self.get_file(os.path.join(MTDSample.BASE_DATASET_PATH, 'data_META', f'MTD{self.id}_*.json'))
    self.wp = self.get_file(os.path.join(MTDSample.BASE_DATASET_PATH, 'data_ALIGNMENT', f'MTD{self.id}_*.csv'))
    self.wav = self.get_file(os.path.join(MTDSample.BASE_DATASET_PATH, 'data_AUDIO', f'MTD{self.id}_*.wav'))

  def get_humdrum(self):
    return open(self.score_humdrum, 'r').read()

  def get_file(self, fn):
      files = glob.glob(fn)
      assert len(files) == 1, '{} does not exist.'.format(fn)
      return files[0]

  def get_spectrogram_db(self):
    audio_time_series = librosa.load(self.wav, mono=True)
    X = librosa.stft(audio_time_series, hop_length=MTDSample.SPECTROGRAM_DISPLAY_HOP_LENGTH, win_length=MTDSample.WIN_LENGTH)
    D = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    return D

  def save_spectogram_into_dataset(self):
    plt.axis('off')
    img = librosa.display.specshow(self.get_spectrogram_db(), x_axis='s',
                             hop_length=MTDSample.SPECTROGRAM_DISPLAY_HOP_LENGTH, y_axis='log')
    img.figure.savefig(self.TEMP_IMG_FILENAME, bbox_inches='tight', pad_inches=0)
    pil_image = PIL.Image.open(self.TEMP_IMG_FILENAME)
    pil_image = pil_image.resize((self.IMG_WIDTH, self.IMG_HEIGHT), resample=PIL.Image.LANCZOS)
    filename = self.wav[:-3]
    filename = filename.replace('data_AUDIO', 'data_AUDIO_IMG')
    filename += "png"
    self.spectogram_img = filename
    pil_image.save(filename)
    subprocess.run(["rm", self.TEMP_IMG_FILENAME])

sample = MTDSample(1127)

# sample.save_spectogram_into_dataset()
# sample.get_humdrum()