#!/usr/bin/python

import os
import numpy as np

class Config:
    def __init__(self,
                 feature_folder,
                 cooccurrence_word_exclude_stopword=False,
                 ):
    
        
        ## path
        self.data_folder = "../../Data"
        self.feature_folder = feature_folder
        self.original_train_data_path = "%s/train.csv" % self.data_folder
        self.original_test_data_path = "%s/test.csv" % self.data_folder
        self.processed_train_data_path = "%s/train.processed.csv" % self.feature_folder
        self.processed_test_data_path = "%s/test.processed.csv" % self.feature_folder
        self.preprocessed_train_data_path='../../Preprocessed_data/train_pre_lower.csv'
        self.preprocessed_train_join_data_path='../../Preprocessed_data/train_join.csv'
        
        ## nlp related        
        self.cooccurrence_word_exclude_stopword = cooccurrence_word_exclude_stopword

        ## create feat folder
        if not os.path.exists(self.feature_folder):
            os.makedirs(self.feature_folder)

        ## creat folder for the training and testing feat
        if not os.path.exists("%s/All" % self.feature_folder):
            os.makedirs("%s/All" % self.feature_folder)    

     
## initialize a param config					
config = Config(feature_folder="../../Feat",cooccurrence_word_exclude_stopword=False)