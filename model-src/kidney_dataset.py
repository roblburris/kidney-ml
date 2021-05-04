import pandas as pd
import scanpy as sc
import numpy as np
import torch
from anndata import AnnData
from torch.utils.data import Dataset 

from process_data import *

class KidneyDataset(Dataset):
    '''
    Gets KidneyDataset to be used in NN for kidney cell classification
    '''
    def __init__(self, adata):
        '''
        pyTorch Dataset Constructor

        Arguments:
            adata: adata object
        '''
        self.adata = AnnData.copy(adata)
        
    def __len__(self):
        '''
        Gets length of dataset

        :return: integer representing length
        '''
        return self.adata.X.shape[0]
    
    def __getitem__(self, idx):
        '''
        Gets item from dataset
        
        :param index: integer representing index
        :return: a tuple where return[0] is the data and
        return[1] is the label
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (
            self.adata.X[idx,:].toarray(), 
            int(self.adata.obs.iloc[idx].name[1]) - 1)
    
    def run_feature_selection(self):
        '''
        Runs feature selection using scanpy's
        highly_variable_genes function
        '''
        sc.pp.highly_variable_genes(self.adata)
        self.adata = self.adata[:, adata.var.highly_variable]

    def get_variable_features():
        '''
        Returns variable features as picked
        by scanpy
        
        Note: assumes run_feature_selection
        has already been called
        '''
        return self.adata.var.highly_variable
        
        

        

    

