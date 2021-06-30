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

import os
import PIL

import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt

class MTDSample:

	DATA_DIR_PREFIX = './MTD/data_'
	TEMP_IMG_FILENAME = 'TMP_IMAGE.png'

	def __init__(self, id : str):
		self.id 					=	id
		self.audio_wav_path 		=	MTDSample._find_filename(self.id, 'AUDIO')
		self.meta_data_path 		=	MTDSample._find_filename(self.id, 'META')
		self.score_humdrum_path 	=	MTDSample._find_filename(self.id, 'SCORE_HUMDRUM')
		try:
			self.audio_spec_path	=	MTDSample._find_filename(self.id, 'AUDIO_IMG')
		except:
			self.audio_spec_path	=	"NO_SPECTROGRAM"

	def _find_filename(MTD_id, directory):
		directory_path = MTDSample.DATA_DIR_PREFIX + directory
		dir_files = os.listdir(directory_path)
		matched_files = [ file for file in dir_files if F"MTD{MTD_id}_" in file ]
		no_matched_files = len(matched_files)
		if no_matched_files == 0:
			raise Exception(F'File with MTD_ID: {MTD_id} could not be found in {directory_path}')
		elif no_matched_files > 1:
			raise Exception(F'File with MTD_ID: {MTD_id} had too many matches ({no_matched_files})')
		else:
			return F'{directory_path}/{matched_files[0]}'

	def get_spectrogram_db(self):
		audio_time_series, achieved_sample_rate = librosa.load(self.audio_wav_path, mono=True)
		X = librosa.stft(audio_time_series, hop_length=PARAM['STFT']['HOP_LENGTH'], win_length=PARAM['STFT']['WIN_LENGTH'])
		D = librosa.amplitude_to_db(np.abs(X), ref=np.max)
		return D

	def get_humdrum_tokens(self):
		return open(self.score_humdrum_path, 'r').read().splitlines()

	def save_spectogram_into_dataset(self):
		plt.axis('off')
		D = self.get_spectrogram_db()
		img = librosa.display.specshow(D, x_axis='linear',
		                        	   hop_length=PARAM['STFT']['HOP_LENGTH'], y_axis=PARAM["SPEC"]["Y_AXIS_SCALE"])
		img.figure.savefig(MTDSample.TEMP_IMG_FILENAME, bbox_inches='tight', pad_inches=0)
		pil_image = PIL.Image.open(MTDSample.TEMP_IMG_FILENAME)
		pil_image = pil_image.resize((D.shape[1] * PARAM['SPEC']['IMG_WIDTH_PER_FRAME'], PARAM['SPEC']['IMG_HEIGHT']), resample=PIL.Image.LANCZOS)
		filename = self.audio_wav_path[:-3]
		filename = filename.replace('data_AUDIO', 'data_AUDIO_IMG')
		filename += 'png'
		pil_image.save(filename)
		self.audio_spec_path = MTDSample._find_filename(self.id, 'AUDIO_IMG')
		os.system(F'rm {MTDSample.TEMP_IMG_FILENAME}')