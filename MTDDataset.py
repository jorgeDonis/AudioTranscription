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

from MTDSample import MTDSample
import os
import re
import numpy as np
import pickle

class HumdrumTranslator:

    def __init__(self, humdrum_dict, humdrum_array):
        self.humdrum_dict = humdrum_dict
        self.humdrum_array = humdrum_array

    def encode_humdrum_token(self, humdrum_token : str) -> int:
        return self.humdrum_dict[humdrum_token]
    
    def decode_humdrum_class_index(self, index : int) -> str:
        return self.humdrum_array[index]

_MTD_id_regex = r"MTD(\d{4}(-\d+)?)"
_HUMDRUM_TRANSLATOR_FILEPATH = './MTD/humdrum_translator.bin'

def get_all_MTD_ids():
    filenames = os.listdir('./MTD/data_AUDIO')
    ids = [ re.findall(_MTD_id_regex, filename)[0][0] for filename in filenames ]
    return sorted(ids)

def export_all_spectrograms():
    for id in get_all_MTD_ids():
        sample = MTDSample(id)
        sample.save_spectogram_into_dataset()

#Generates a dictionary { 'humdrum_token' : humdrum_class_index }
def gen_humdrum_dict():
    dict = {}
    for id in get_all_MTD_ids():
        sample = MTDSample(id)
        new_tokens = sample.get_humdrum_tokens()
        for token in new_tokens:
            if token not in dict:
                dict[token] = len(dict)
    return dict

#Generates a numpy array [ [0] => 'first_humdrum_token', [1] => 'second_humdrum_token' ... ]
def gen_humdrum_array(humdrum_dict):
    return np.array(list(humdrum_dict.keys()))

def gen_humdrum_translator()-> HumdrumTranslator:
    dict = gen_humdrum_dict()
    array = gen_humdrum_array(dict)
    translator = HumdrumTranslator(dict, array)
    with open(_HUMDRUM_TRANSLATOR_FILEPATH, 'wb') as file:
        pickle.dump(translator, file, protocol=pickle.HIGHEST_PROTOCOL)
        return translator

def get_humdrum_translator() -> HumdrumTranslator:
    if not os.path.isfile(_HUMDRUM_TRANSLATOR_FILEPATH):
        return gen_humdrum_translator()
    else:
        with open(_HUMDRUM_TRANSLATOR_FILEPATH, 'rb') as file:
            return pickle.load(file)