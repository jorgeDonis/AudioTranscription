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

import MTDSample

import numpy as np
import cv2

#Spectrogram images must be generated previously
#Returns a list of tuples [ [padded_img, padded_humdrum, true_img_len, true_humdrum_len ], [ ... ] ... ]
def gen_batch(MTD_ids: list) -> list:
    samples = _get_samples(MTD_ids)
    original_imgs = _get_original_imgs(samples)
    max_image_len = _get_max_img_len(original_imgs)
    max_humdrum_len = _get_max_humdrum_len(samples)
    return []


def _get_max_humdrum_len(MTD_samples: list) -> int:
    max_humdrum_len = -1
    for sample in MTD_samples:
        max_humdrum_len = max(len(sample.get_humdrum(), max_humdrum_len))
    return max_humdrum_len

    
def _get_max_img_len(original_imgs: list) -> int:
    max_len = -1
    for image in original_imgs:
        max_len = max(image.shape[1], max_len)
    return max_len

def _get_original_imgs(MTD_samples : list) -> list:
    imgs = []
    for sample in MTD_samples:
        try:
            np_img = cv2.imread(sample.spectogram_img)
            imgs.append(np_img)
        except:
            raise Exception(F"Error reading image from sample {sample.id}")
    return imgs

def _get_samples(MTD_ids: list) -> list:
    samples = []
    for id in MTD_ids:
        samples.append(MTDSample(id))
    return samples