#!/usr/bin/python

import csv
from autocorrect import spell
from  nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup

def preprocess_train():
	train_data = open('./train.csv')
	csv_train = csv.reader(train_data,delimiter=',')
	out_file = open('./train_pre.csv','w')
	csv_file = csv.writer(out_file,delimiter=',')
	next(csv_train,None)
	for row in csv_train:
		corrected_string = spell_correct(row[3])
		row.append(corrected_string)
		row[2] = stem_term(row[2])
		row[3] = stem_term(row[3])
		row[5] = stem_term(row[5])		
		csv_file.writerow(row)
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

def process_description():
	desc = open('product_description.csv')
	desc_pre = open('product_description_pre.csv','w')
	csv_desc_pre = csv.writer(desc_pre,delimiter=',')
	csv_desc = csv.reader(desc,delimiter=',')
	next(csv_desc,None)
	for row in csv_desc:
		row[1] = drop_html(row[1])
		row[1] = stem_term(row[1])
		csv_desc_pre.writerow(row)
	close(desc)
	close(desc_pre)



###################
## Drop html tag ##
###################
def drop_html(html):
    return BeautifulSoup(html).get_text(separator=" ")


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
	preprocess_train()
	process_description()