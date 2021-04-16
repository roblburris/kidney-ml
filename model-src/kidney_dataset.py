import pandas as pd
import scanpy as sc
import numpy as np
import torch
from torch.utils.data import Dataset 

class KidneyDataset(Dataset):
    '''
    Gets KidneyDataset to be used in NN for kidney cell classification
    '''
    def __init__(self, path):
        '''
        pyTorch Dataset Constructor

        :param path: filepath to the directory containing the 10x-Genomics mtx sparse
        matrix
        '''
        self.adata = self.__get_adata(path)
        
    def __get_adata(self, path):
        '''
        Helper function that creates a new adata object (see scanpy)
        
        :param path: path to the directory containing mtx file
        '''
        adata = sc.read(path + 'matrix.mtx', cache=True)
        adata.obs_names = pd.read_csv(path + 'genes.tsv', header=None, sep='\t')[0]
        adata.var_names = pd.read_csv(path + 'barcodes.tsv', header=None, sep='\t')[0]
        return adata

    def __len__(self):
        '''
        Gets length of dataset

        :return: integer representing length
        '''
        return self.adata.X.shape[1]
    
    def __getitem__(self, idx):
        '''
        Gets item from dataset
        
        :param index: integer representing index
        :return: a tuple where return[0] is the data and
        return[1] is the label
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return (self.adata.X[:,idx].toarray(), int(self.adata.var.iloc[idx].name[1]) - 1)

        
        

        

    

