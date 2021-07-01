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

import cv2
import numpy as np

def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img = np.expand_dims(img, axis=2)
    img = img / 255
    return img

#Pads with black pixels (0's)
def pad_img_horizontal(img, max_img_len):
    return np.pad(img, ( (0, 0), (0, max_img_len - img.shape[1]), (0, 0) ), 'constant', constant_values= ( (0, 0), (0, 0), (0, 0) ))