import os
from os import system
import requests
import random

def __download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = __get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    __save_response_content(response, destination)    

def __get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def __save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

COMPRESSED_FILENAME = "OCR_dataset.tar.gz"
GOOGLE_DRIVE_FILE_ID = "1Uhnm-n8AFx1mfmdwgzaK4BIPJo8l0CCh"
DATASET_DIR = "OCR_dataset_2/"
DATASET_DIR_TRAIN = "OCR_dataset_train/"
DATASET_DIR_TEST = "OCR_dataset_test/"
NO_TRAINING_SAMPLES = 20000
NO_TEST_SAMPLES = 1000


def _check_download_dataset():
    if not os.path.isdir(DATASET_DIR) or len(os.listdir(DATASET_DIR)) < (NO_TRAINING_SAMPLES + NO_TEST_SAMPLES):
        if not os.path.isfile(COMPRESSED_FILENAME):
            __download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, COMPRESSED_FILENAME)
        system("tar -xvzf " + COMPRESSED_FILENAME)
        system("rm -f " + COMPRESSED_FILENAME)

def _check_create_train_test_dirs():
    if not os.path.isdir(DATASET_DIR_TRAIN):
        system("mkdir " + DATASET_DIR_TRAIN)
    if not os.path.isdir(DATASET_DIR_TEST):
        system("mkdir " + DATASET_DIR_TEST)
    if len(os.listdir(DATASET_DIR_TRAIN)) != NO_TRAINING_SAMPLES:
        system("rm " + DATASET_DIR_TRAIN + "*")
    if len(os.listdir(DATASET_DIR_TEST)) != NO_TEST_SAMPLES:
        system("rm " + DATASET_DIR_TEST + "*")

def _check_copy_split_samples():
    filenames = sorted(os.listdir(DATASET_DIR))
    if len(os.listdir(DATASET_DIR_TRAIN)) == 0:
        filenames_train = [ filenames.pop(random.randrange(len(filenames))) for filename in range(NO_TRAINING_SAMPLES) ]
        for filename in filenames_train:
            print(F"copying {DATASET_DIR}{filename} => {DATASET_DIR_TRAIN}{filename}")
            os.system(F"cp {DATASET_DIR}{filename} {DATASET_DIR_TRAIN}{filename}")
    if len(os.listdir(DATASET_DIR_TEST)) == 0:
        filenames_test = [ filenames.pop(random.randrange(len(filenames))) for filename in range(NO_TEST_SAMPLES) ]
        for filename in filenames_test:
            print(F"copying {DATASET_DIR}{filename} => {DATASET_DIR_TEST}{filename}")
            os.system(F"cp {DATASET_DIR}{filename} {DATASET_DIR_TEST}{filename}")


def init():
    _check_download_dataset()
    _check_create_train_test_dirs()
    _check_copy_split_samples()

def get_val_ds_filenames():
    filenames = os.listdir(DATASET_DIR_TEST)
    filepaths = [ DATASET_DIR_TEST + filename for filename in filenames ]
    return filepaths
