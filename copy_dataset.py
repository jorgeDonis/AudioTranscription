import os
import random

N_TRAIN = 10000
N_TEST = 1000

filenames = sorted(os.listdir("OCR_dataset_2/"))
filenames_train = [ filenames.pop(random.randrange(len(filenames))) for filename in range(N_TRAIN) ]
filenames_test = [ filenames.pop(random.randrange(len(filenames))) for filename in range(N_TEST) ]

for filename in filenames_train:
    print("cp ./OCR_dataset_2/" + filename + " ./OCR_dataset_3_train")
    os.system("cp ./OCR_dataset_2/" + filename + " ./OCR_dataset_3_train")

for filename in filenames_test:
    print("cp ./OCR_dataset_2/" + filename + " ./OCR_dataset_3_test")
    os.system("cp ./OCR_dataset_2/" + filename + " ./OCR_dataset_3_train")