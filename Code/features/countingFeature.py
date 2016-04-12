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
from utils import try_divide

spell_check_dict=dict()

def readSpellDict():
	fileName="spell_dict"
	f=open(fileName, 'r')
	for line in f:
		line=line[:-1]
		s=line.split(": ")
		spell_check_dict[s[0]]=s[1]

def get_position_list(target, obs):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target

##To read a CSV and return list of list####
def get_csv_data(fileName):
	reader = pd.read_csv(fileName)
	#next(reader,None)
	return reader	


def preprocess_data(line):
    line=str(line)
    if line in spell_dict:
    	line=spell_dict[line]
    token_pattern = r"(?u)\b\w\w+\b"
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    tokens = [x.lower() for x in token_pattern.findall(line)]
    return tokens


def extract_feat(df):
	join_str="_"
	df["query_unigram"] = list(df.apply(lambda x: preprocess_data(x["query_original"]), axis=1))
	df["title_unigram"] = list(df.apply(lambda x: preprocess_data(x["title"]), axis=1))
	df["description_unigram"] = list(df.apply(lambda x: preprocess_data(x["description"]), axis=1))
	df["attribute_values_unigram"] = list(df.apply(lambda x: preprocess_data(x["attribute_values"]), axis=1))
	df["query_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["query_unigram"], join_str), axis=1))
	df["title_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["title_unigram"], join_str), axis=1))
	df["description_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["description_unigram"], join_str), axis=1))
	df["attribute_values_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["attribute_values_unigram"], join_str), axis=1))
	df["query_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["query_unigram"], join_str), axis=1))
	df["title_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["title_unigram"], join_str), axis=1))
	df["description_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["description_unigram"], join_str), axis=1))
	df["attribute_values_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["attribute_values_unigram"], join_str), axis=1))

	################################
	## word count and digit count ##
	################################
	print "generate word counting features"
	feat_names = ["query", "title", "description", "attribute_values"]
	grams = ["unigram", "bigram", "trigram"]
	count_digit = lambda x: sum([1. for w in x if w.isdigit()])
	for feat_name in feat_names:
		for gram in grams:
			## word count
			df["count_of_%s_%s"%(feat_name,gram)] = list(df.apply(lambda x: len(x[feat_name+"_"+gram]), axis=1))
			df["count_of_unique_%s_%s"%(feat_name,gram)] = list(df.apply(lambda x: len(set(x[feat_name+"_"+gram])), axis=1))
			df["ratio_of_unique_%s_%s"%(feat_name,gram)] = map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)])

		## digit count
		df["count_of_digit_in_%s"%feat_name] = list(df.apply(lambda x: count_digit(x[feat_name+"_unigram"]), axis=1))
		df["ratio_of_digit_in_%s"%feat_name] = map(try_divide, df["count_of_digit_in_%s"%feat_name], df["count_of_%s_unigram"%(feat_name)])

	## description missing indicator
	df["description_missing"] = list(df.apply(lambda x: int(x["description_unigram"] == ""), axis=1))


    ##############################
    ## intersect word count ##
    ##############################

	print "generate intersect word counting features"
	#### unigram
	for gram in grams:
		for obs_name in feat_names:
			for target_name in feat_names:
				if target_name != obs_name:
                    ## query
					df["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = list(df.apply(lambda x: sum([1. for w in x[obs_name+"_"+gram] if w in set(x[target_name+"_"+gram])]), axis=1))
					df["ratio_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = map(try_divide, df["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)], df["count_of_%s_%s"%(obs_name,gram)])

		## some other feat
		df["title_%s_in_query_div_query_%s"%(gram,gram)] = map(try_divide, df["count_of_title_%s_in_query"%gram], df["count_of_query_%s"%gram])
		df["title_%s_in_query_div_query_%s_in_title"%(gram,gram)] = map(try_divide, df["count_of_title_%s_in_query"%gram], df["count_of_query_%s_in_title"%gram])
		df["description_%s_in_query_div_query_%s"%(gram,gram)] = map(try_divide, df["count_of_description_%s_in_query"%gram], df["count_of_query_%s"%gram])
		df["description_%s_in_query_div_query_%s_in_description"%(gram,gram)] = map(try_divide, df["count_of_description_%s_in_query"%gram], df["count_of_query_%s_in_description"%gram])




	######################################
	## intersect word position feat ##
	######################################




	print "generate intersect word position features"
	for gram in grams:
		for target_name in feat_names:
			for obs_name in feat_names:
				if target_name != obs_name:
					pos = list(df.apply(lambda x: get_position_list(x[target_name+"_"+gram], obs=x[obs_name+"_"+gram]), axis=1))
					## stats feat on pos
					df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(np.min, pos)
					df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(np.mean, pos)
					df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(np.median, pos)
					df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(np.max, pos)
					df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(np.std, pos)
					## stats feat on normalized_pos
					df["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
					df["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
					df["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
					df["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
					df["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] , df["count_of_%s_%s" % (obs_name, gram)])


	print "dropping unigrams bigrams trigrams"
	df=df.drop(['query_unigram', 'title_unigram', 'description_unigram', 'query_bigram','title_bigram','description_bigram', 'query_trigram', 'title_trigram', 'description_trigram'], axis=1)                      	
	print "creating csv"
	df.to_csv("../../data/trial/train_distFeat_counting_updated.csv", header=True, index=False)

if __name__ == "__main__":

	readSpellDict()
	input_df=get_csv_data("../../data/trial/train_distFeat_updated.csv")
	input_df=input_df.replace(np.nan,0, regex=True)

	#######################
	## Generate Features ##
	#######################
	extract_feat(input_df)
	

	
