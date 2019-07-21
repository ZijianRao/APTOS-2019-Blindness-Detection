import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import hashlib
from os.path import isfile
from joblib import Parallel, delayed
import psutil

import config

def clean_data():
    train = pd.read_csv(config.LOCAL_TRAIN_DATA_PATH)
    img_meta_l = Parallel(n_jobs=psutil.cpu_count(), verbose=1)(
    (delayed(get_image_meta_data)(fp) for fp in train.id_code))
    img_meta_df = pd.DataFrame(np.array(img_meta_l))
    img_meta_df.columns = ['id_code', 'strMd5']
    train = train.merge(img_meta_df, on='id_code')
    # leave one consistent
    train = train.drop_duplicates(subset=['strMd5', 'diagnosis'], keep='last')
    # remove inconsistent cases
    train = train.drop_duplicates(subset=['strMd5'], keep=False)
    train.to_csv('../data/clean_sample.csv', index=None)


def get_image_meta_data(p):
    """Get meta to calculate md5
    https://www.kaggle.com/h4211819/more-information-about-duplicate
    Args:
        p ([type]): [description]
    
    Returns:
        [type]: [description]
    """

    strFile = config.TRAIN_IMAGE_PATH.format(p)
    file = None
    bRet = False
    strMd5 = ""
    
    try:
        file = open(strFile, "rb")
        md5 = hashlib.md5()
        strRead = ""
        
        while True:
            strRead = file.read(8096)
            if not strRead:
                break
            md5.update(strRead)
        #read file finish
        bRet = True
        strMd5 = md5.hexdigest()
    except:
        bRet = False
    finally:
        if file:
            file.close()

    return p, strMd5

if __name__ == '__main__':
    clean_data()