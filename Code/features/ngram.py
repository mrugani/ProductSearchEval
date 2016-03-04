import pandas as pd

def getUnigram(words):
	return words

def getBigram(words, join="_"):
	l=len(words)
	result=[]
	if(l<=1):
		return words
	for i in range(l-1):
		result.append(words[i]+join+words[i+1])

	return result

def getTrigram(words, join="_"):
	l=len(words)
	result=[]
	if(l<=2):
		return words
	for i in range(l-2):
		result.append(words[i]+join+words[i+1]+join+words[i+2])

	return result


