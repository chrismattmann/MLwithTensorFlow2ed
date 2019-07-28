from tqdm.auto import tqdm as tqdm_nn
import pandas as pd
import numpy as np
from multiprocessing import Pool
import requests
from requests.exceptions import ConnectionError
from requests.exceptions import ReadTimeout

print('Reading Dataframe.')
df = pd.read_csv('vgg_face_full.csv')

tqdm_nn.pandas()
timeout=1

def url_ok(url):
    try:
        r = requests.head(url, timeout=timeout)
        return r.status_code == 200
    except (ConnectionError, ReadTimeout)  as e:
        #print("URL connection error", url)
        return False

def fx(df):
    df['VALID_URL'] = df['URL'].progress_apply(url_ok)
    return df

def parallelize_dataframe(df, func, n_cores=8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

df = parallelize_dataframe(df, fx)
print('Writing Dataframe.')
df.to_csv('vgg_face_full_urls.csv')
