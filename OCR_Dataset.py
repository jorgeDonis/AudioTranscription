import os
from os import system
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

class OCR_Dataset:

    COMPRESSED_FILENAME = "OCR_dataset.tar.gz"
    GOOGLE_DRIVE_FILE_ID = "1Uhnm-n8AFx1mfmdwgzaK4BIPJo8l0CCh"
    DATASET_DIR = "OCR_dataset_2/"
    DEBUG_DATASET_DIR_TRAIN = "OCR_dataset_3_train/"

    def __init__(self):
        if not os.path.isdir(OCR_Dataset.DATASET_DIR):
            if not os.path.isfile(OCR_Dataset.COMPRESSED_FILENAME):
                download_file_from_google_drive(OCR_Dataset.GOOGLE_DRIVE_FILE_ID, OCR_Dataset.COMPRESSED_FILENAME)
            system("tar -xvzf " + OCR_Dataset.COMPRESSED_FILENAME)
            system("rm -f " + OCR_Dataset.COMPRESSED_FILENAME)