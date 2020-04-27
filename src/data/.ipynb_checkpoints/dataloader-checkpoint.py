"""
Project: Object Recognising Traffic Signs Using Deep Learning
Mentor: Prof. Ankur Sinha


@author: Ashish Panchal(aashish.panchal@gmail.com)
"""


import requests
import click
import os
import os.path
from urllib.parse import urlparse
import pathlib
import pickle

train_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
test_GT_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
test_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"


RAW_DATA_PATH="../data/raw/"
INTERIM_DATA_PATH= "../data/interim/"
PROCESSED_DATA_PATH= "../data/processed/"
ADDITIONAL_SIGNS_PATH = "..data/traffic_sign_images/"


URLS= [train_URL, test_URL, test_GT_URL]
pickled_files = ['train.p','test.p', 'valid.p']


def convert_raw_to_interim():
    print('converting raw data into interim data format [pickled]')



def load_interim_data():
    """
    load interim dataset
    """
    interim_ds = []
    for f in pickled_files:
        filename = os.path.abspath(INTERIM_DATA_PATH+f)
        print('Loading {}'.format(filename))
        with open(filename, mode='rb') as f:
            interim_ds.append(pickle.load(f))
    return interim_ds




def download_data():
    for url in URLS:
        print('Checking for existance of {0}'.format(url))
        a = urlparse(url)
        filename = os.path.basename(a.path)
        data_path_name = os.path.abspath(RAW_DATA_PATH)
        filename = data_path_name+os.path.sep+filename
        f = pathlib.Path(filename)
        if f.exists:
            print(filename, ' is already exists.')
        else:
            download_file(url,filename)
        

def download_file(url, filename):
    print('Downloading from {} to {}'.format(url,filename))
    response = requests.get(url)
    with open(filename, 'wb') as ofile:
        ofile.write(response.content)


