import numpy as np
import cv2
from multiprocessing import Pool
from skimage import color
from skimage.io import imread
import pandas as pd
import os
from tqdm import tqdm_notebook as tqdm
from tqdm.auto import tqdm as tqdm_nn
import random
import requests
from urllib.error import HTTPError, URLError
from requests.exceptions import ConnectionError
from requests.exceptions import ReadTimeout
from http.client import IncompleteRead

def fx(df):
    for index, row in tqdm(df.iterrows()):
        url = row['URL']
        box = row[['LEFT', 'TOP', 'RIGHT', 'BOTTOM']].values
        celeb = row['CELEB']
        number = row['ID']
        clip_image_and_save(url, box, celeb, number)    


def parallelize_dataframe(df, func, n_cores=48):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    pool.map(func, df_split)
    pool.close()
    pool.join()

def clip_image_and_save(url, bbox, celeb, number):

    directory = 'vgg-face/'+celeb
    img_name = directory+'/'+str(number)+'_244x244.png'
    
    if os.path.exists(img_name):
        print('Image '+img_name+' already exists, skipping.')
        return
    
    try:
        image = imread(url)
    except (AttributeError, HTTPError, ConnectionResetError, ConnectionRefusedError, URLError, ValueError, IncompleteRead) as e:
        print('Error writing url: '+url+' skipping. Message: '+str(e))
        return
    
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    
    w=x2-x1
    h=y2-y1
    

    if not os.path.exists(directory):
        print('Making '+directory)
        os.makedirs(directory)
    
    crop_img = image[y1:y1+h, x1:x1+w]
    new_size = 224,224
    try:
        crop_img = cv2.resize(crop_img, new_size, interpolation=cv2.INTER_CUBIC)
    except  cv2.error as e:
        print('Error cropping with CV2 for image: '+url+'. skipping. Message: '+str(e))
        return
    
    print('Writing '+img_name)
    cv2.imwrite(img_name,crop_img)
    
tqdm_nn.pandas()
valid_face_urls_path = 'vgg_face_full_urls.csv'
print('Reading Dataframe.')
df = pd.read_csv(valid_face_urls_path) 
df = df[df.VALID_URL==True]
print('Reducing Dataframe to valid URLs and resetting index.')
df.reset_index(inplace=True)

parallelize_dataframe(df, fx, 12)


