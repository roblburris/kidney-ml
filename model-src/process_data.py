import scanpy as sc
import pandas as pd
import math, anndata, random
from scipy.io import mmread

def load_data(path):
    '''
    Loads data from files and creates train, test
    sets

    Arguments:
        path: path to directory containing
              sc-RNA seq data files
    
    Returns:
        anndata object
    '''
    adata = sc.read_h5ad('./cache/data-mouse-adult-matrix.h5ad')
    adata = adata.transpose()
    adata.obs_names = pd.read_csv(path + 'barcodes.tsv', header=None, sep='\t')[0]
    adata.var_names = pd.read_csv(path + 'genes.tsv', header=None, sep='\t')[0]

    return adata 

def train_test(adata, train_size):
    '''
    Creates train and test sets based on split
    size

    Arguments:
        adata: anndata object (base dataset)
        train_size: float representing proportion of dataset
        to be used in train set
    Returns:
        a tuple (train_set, test_set) of anndata objects
    '''
    train_size = math.floor(train_size * adata.X.shape[0])

    observations = set(adata.obs.index.tolist())
    train_obs = random.sample(observations, train_size)
    test_obs = list(observations.difference(train_obs))
    train_obs = list(train_obs)

    test_set = anndata.AnnData.copy(adata)
    test_set = test_set[test_obs, :]
    adata = adata[train_obs, :]

    return (adata, test_set)