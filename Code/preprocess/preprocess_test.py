#!/usr/bin/python

import csv
from autocorrect import spell
from  nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def remove_non_ascii_1(text):
    return ''.join(i for i in text if ord(i)<128)

def preprocess_test():
	test_data = open('../../../test.csv')
	csv_test = csv.reader(test_data,delimiter=',')
	out_file = open('../../../test_pre.csv','w')
	csv_file = csv.writer(out_file,delimiter=',')
	next(csv_test,None)
	cnt = 1;
	for row in csv_test:
		corrected_string = spell_correct(remove_non_ascii_1(row[3])).lower()
		row.append(corrected_string)
		row[2] = stem_term(remove_non_ascii_1(row[2])).lower()
		row[3] = stem_term(remove_non_ascii_1(row[3])).lower()
		row[4] = stem_term(remove_non_ascii_1(row[4])).lower()		
		csv_file.writerow(row)
		print cnt
		cnt+=1
	close(out_file)
	close(train_data)


def stem_term(item):
	ps = PorterStemmer()
	item = item.split()
	stemmed_string = ''
	for term in item:
		stemmed_string += ps.stem(term)+' '
	stemmed_string = stemmed_string.strip()
	return stemmed_string



def spell_correct(search_term):
		search_term  = search_term.split()
		corrected_string = ''
		for term in search_term:
			corrected_term = term
			if term.isalpha():
				corrected_term = spell(term)
			corrected_string += corrected_term+' '
		return corrected_string.strip()
		

if __name__ == '__main__':
	preprocess_test()