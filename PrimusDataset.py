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

from Parameters import Parameters as PARAM
from PrimusSample import PrimusSample
import ImageProcessing

from typing import Generator, List, Tuple
import os
import numpy as np
import pickle
import random

_SEMANTIC_TRANSLATOR_FILEPATH = './semantic_translator.bin'
_DATASET_IDS_BINARY = 'train_test_ids.bin'

class SemanticTranslator:

    def __init__(self, semantic_dict, semantic_array):
        self.semantic_dict = semantic_dict
        self.semantic_array = semantic_array
        self.blank_class = len(semantic_array)

    def encode_semantic_token(self, semantic_token : str) -> int:
        return self.semantic_dict[semantic_token]
    
    def decode_semantic_class_index(self, index : int) -> str:
        return self.semantic_array[index]

    def encode_semantic_token_seq(self, token_seq) -> List:
        return [ self.encode_semantic_token(token) for token in token_seq ]

    def decode_semantic_class_index_seq(self, class_index_seq) -> List:
        return [ self.decode_semantic_class_index(class_index) for class_index in class_index_seq ]

def get_all_primus_ids():
    return sorted(os.listdir('./Complete_Primus'))

def all_primus_samples():
    for id in get_all_primus_ids():
        yield PrimusSample(id)

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

def get_train_test_ids():
    with open(_DATASET_IDS_BINARY, 'rb') as file:
        return pickle.load(file)

def num_train_samples():
    train_ids, test_ids = get_train_test_ids()
    return len(train_ids)

#generates binary file which is used in generators
def gen_train_test_ids(no_train_samples, no_test_samples):
    ids = os.listdir("./Complete_Primus")
    if no_train_samples + no_test_samples > len(ids):
        raise Exception(F'Dataset has only {len(no_train_samples)} total samples')
    ids_train = [ ids.pop(random.randrange(len(ids))) for id in range(no_train_samples) ]
    ids_test = [ ids.pop(random.randrange(len(ids))) for id in range(no_test_samples) ]
    with open(_DATASET_IDS_BINARY, 'wb') as file:
        pickle.dump((ids_train, ids_test), file)

def _num_repeated_tokens(class_index_seq):
    num_repeated_tokens = 0
    if len(class_index_seq) == 0:
        return num_repeated_tokens
    previous_token = class_index_seq[0]
    for i in range(1, len(class_index_seq)):
        if class_index_seq[i] == previous_token:
            num_repeated_tokens += 1
        previous_token = class_index_seq[i]
    return num_repeated_tokens

def _images_wide_enough(imgs, symbolic_seqs):
    for img, symbolic_seq in zip(imgs, symbolic_seqs):
        if img.shape[1] // PARAM['TRAINING']['POOLING_RATIO'] < (len(symbolic_seq) + _num_repeated_tokens(symbolic_seq)):
            return False
    return True

def _pad_encoding(encoding, new_length, blank_token):
    for i in range(0, new_length - len(encoding)):
        encoding.append(blank_token)
    return encoding

def _gen_train_batch(images, encodings, blank_token):
    max_img_len = max([ x.shape[1] for x in images ])
    max_encoding_len = max([ len(x) for x in encodings ])
    original_image_lengths_after_pooling = np.array([ x.shape[1] // PARAM['TRAINING']['POOLING_RATIO'] for x in images ])
    original_encoding_lengths = np.array([ len(x) for x in encodings ])
    images = [ ImageProcessing.pad_img_horizontal(x, max_img_len) for x in images ]
    encodings = [ _pad_encoding(x, max_encoding_len, blank_token) for x in encodings ]
    return {
        'padded_images' : np.array(images),
        'padded_encodings' : np.array(encodings),
        'original_image_lengths_after_pooling' : original_image_lengths_after_pooling,
        'original_encoding_lengths' : original_encoding_lengths
    }

def _gen_val_batch(images):
    max_img_len = max([ x.shape[1] for x in images ])
    images = [ ImageProcessing.pad_img_horizontal(x, max_img_len) for x in images ]
    return { 'padded_images' : np.array(images) }

#Infinite generator
def train_generator():
    train_ids, test_ids = get_train_test_ids()
    semantic_translator = get_semantic_translator()
    i = 0
    images = []
    encodings = []
    while True:
        for id in train_ids:
            sample = PrimusSample(id)
            img = sample.get_preprocesssed_img()
            images.append(img)
            encodings.append(semantic_translator.encode_semantic_token_seq(sample.get_semantic_tokens()))
            i += 1
            if i == PARAM['TRAINING']['BATCH_SIZE']:
                if _images_wide_enough(images, encodings):
                    inputs_fit = _gen_train_batch(images, encodings, semantic_translator.blank_class)
                    outputs_fit = { 'ctc' : np.zeros([PARAM['TRAINING']['BATCH_SIZE']]) }
                    yield inputs_fit, outputs_fit
                images.clear()
                encodings.clear()
                i = 0


def val_generator_factory() -> Generator:
    return validation_generator()

def validation_generator():
    semantic_translator = get_semantic_translator()
    train_ids, test_ids = get_train_test_ids()
    i = 0
    images = []
    encodings = []
    for id in test_ids:
        sample = PrimusSample(id)
        img = sample.get_preprocesssed_img()
        images.append(img)
        encodings.append(semantic_translator.encode_semantic_token_seq(sample.get_semantic_tokens()))
        i += 1
        if i == PARAM['TRAINING']['BATCH_SIZE']:
            if _images_wide_enough(images, encodings):
                input_val = _gen_val_batch(images)
                yield (input_val, encodings)
            images.clear()
            encodings.clear()
            i = 0

def get_random_sample() -> PrimusSample:
    rand_id = random.choice(get_all_primus_ids())
    return PrimusSample(rand_id)
