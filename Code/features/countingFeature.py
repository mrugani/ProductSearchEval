#!/usr/bin/python

import sys
import ngram
import re
import numpy as np
import csv
import pandas as pd
import ngram  as ng
sys.path.append("../")
from config import config

##To read a CSV and return list of list####
def get_csv_data(fileName):
	reader = pd.read_csv(fileName)
	#next(reader,None)
	return reader	

def preprocess_data(line):
    token_pattern = r"(?u)\b\w\w+\b"
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    tokens = [x.lower() for x in token_pattern.findall(line)]
    return tokens

def extract_feat(df):
	join_str="_"
	df["query_unigram"] = list(df.apply(lambda x: preprocess_data(x["query"]), axis=1))
	df["title_unigram"] = list(df.apply(lambda x: preprocess_data(x["title"]), axis=1))
	df["description_unigram"] = list(df.apply(lambda x: preprocess_data(x["description"]), axis=1))
	df["query_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["query_unigram"], join_str), axis=1))
	df["title_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["title_unigram"], join_str), axis=1))
	df["description_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["description_unigram"], join_str), axis=1))
	df["query_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["query_unigram"], join_str), axis=1))
	df["title_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["title_unigram"], join_str), axis=1))
	df["description_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["description_unigram"], join_str), axis=1))

	print df["query_unigram"]

if __name__ == "__main__":

	input_df=get_csv_data(config.preprocessed_train_join_data_path)

	#######################
    ## Generate Features ##
    #######################
	extract_feat(input_df)
	

	
