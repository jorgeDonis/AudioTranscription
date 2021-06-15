
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

import numpy as np
import cv2
from MTDSample import MTDSample
import BatchGenerator

def show_img(img) -> None:
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

MTD_ids = ['0961', '3795', '4893', '5165', '5853', '6157', '6200', '6237', '6242', '9525']

# for _id in MTD_ids:
#     sample = MTDSample(_id)
#     sample.save_spectogram_into_dataset()

padded_imgs, padded_humdrums, original_img_lengths, original_hum_lengths = BatchGenerator.gen_batch(MTD_ids)

