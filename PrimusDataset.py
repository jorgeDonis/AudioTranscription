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

from typing import List, Tuple
from PrimusSample import PrimusSample
import os
import re
import numpy as np
import pickle

class SemanticTranslator:

    def __init__(self, semantic_dict, semantic_array):
        self.semantic_dict = semantic_dict
        self.semantic_array = semantic_array

    def encode_semantic_token(self, semantic_token : str) -> int:
        return self.semantic_dict[semantic_token]
    
    def decode_semantic_class_index(self, index : int) -> str:
        return self.semantic_array[index]

_SEMANTIC_TRANSLATOR_FILEPATH = './semantic_translator.bin'

def get_all_primus_ids():
    return sorted(os.listdir('./Complete_Primus'))

def export_all_spectrograms(first_index = 0):
    ids = get_all_primus_ids()
    total_to_export = len(ids)
    i = 0
    for id in ids:
        i += 1
        if i > first_index:
            sample = PrimusSample(id)
            print(F'Exporting spectrogram {id} ({i} from {total_to_export})')
            sample.save_spectogram_into_dataset()

#Generates a dictionary { 'semantic_token' : semantic_class_index }
def gen_semantic_dict():
    dict = {}
    for id in get_all_primus_ids():
        sample = PrimusSample(id)
        new_tokens = sample.get_semantic_tokens()
        for token in new_tokens:
            if token not in dict:
                dict[token] = len(dict)
    return dict

def gen_all_wavs(first_index = 0):
    i = 0
    for sample_id in get_all_primus_ids():
        i += 1
        if i > first_index:
            sample = PrimusSample(sample_id)
            print(F"Generating wav file {sample_id}.wav")
            os.system(F'timidity --sampling-freq=44.1 -A 120, 100 -Ow --output-mono '
                      F'-o ./Complete_Primus/{sample_id}/{sample_id}.wav '
                      F'{sample.score_midi}')

#Generates a numpy array [ [0] => 'first_semantic_token', [1] => 'second_semantic_token' ... ]
def gen_semantic_array(semantic_dict):
    return np.array(list(semantic_dict.keys()))

def gen_semantic_translator()-> SemanticTranslator:
    dict = gen_semantic_dict()
    array = gen_semantic_array(dict)
    translator = SemanticTranslator(dict, array)
    with open(_SEMANTIC_TRANSLATOR_FILEPATH, 'wb') as file:
        pickle.dump(translator, file, protocol=pickle.HIGHEST_PROTOCOL)
        return translator

def get_semantic_translator() -> SemanticTranslator:
    if not os.path.isfile(_SEMANTIC_TRANSLATOR_FILEPATH):
        return gen_semantic_translator()
    else:
        with open(_SEMANTIC_TRANSLATOR_FILEPATH, 'rb') as file:
            return pickle.load(file)


train_ids, test_ids = None
with open('dataset_train_test_ids.bin') as file:
    train_ids, test_ids = pickle.load(file)

def train_generator():
    i = 0
    images = []
    labels = []
    while True:
        for id in train_ids:
            