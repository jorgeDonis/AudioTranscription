
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

import PrimusDataset
import random
import pickle

ids_with_img = []

i = 0
for id in PrimusDataset.get_all_primus_ids():
    i += 1
    if i <= 25005:
        ids_with_img.append(id)

test_samples = 1000
training_samples = 10000

ids_test = [ ids_with_img.pop(random.randrange(len(ids_with_img))) for id in range(test_samples) ]
ids_train = [ ids_with_img.pop(random.randrange(len(ids_with_img))) for id in range(training_samples) ]

split_ids = (ids_train, ids_test)

with open('dataset_train_test_ids.bin', 'wb') as file:
    pickle.dump(split_ids, file, protocol=pickle.HIGHEST_PROTOCOL)




