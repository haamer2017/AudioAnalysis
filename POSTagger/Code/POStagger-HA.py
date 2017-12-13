
'''
author : Hafsah Aamer
created on: 4/12/2017

The purpose of this code is to read text and extract pos_tags, lemmas and entities from it, into three files
1.POStags_Lemma_Spacy.csv
2.Entities_Spacy.cvs
3.Entities_nltk.csv

Drawbacks with nltk is that it ra=tags a single entity in different text as Person and Org. Example: Deloitte, Samsung.

#NEED TWO FOLDERS 
1.DATA-- that includes the file to be postagged
2.OUTPUT -- where the files are output
'''

import nltk
import glob
import pandas as pd       
import os 
import math 
import numpy as np
from nltk.corpus import treebank
from nltk.tag.stanford import StanfordNERTagger
from nltk import sent_tokenize,word_tokenize, pos_tag, ne_chunk
from itertools import groupby
import csv
import spacy
import re, string, timeit

def extract_entities_spacy(text):
	
	nlp = spacy.load('en_core_web_sm')
	with open('POStags_Lemma_Spacy.csv', 'w', newline='') as outfile1:
		writer1 = csv.writer(outfile1)
		writer1.writerow(["Token", "Token_lemma_", "pos_tag (Spacy)"])
		with open('Entities_Spacy.csv', 'w', newline='') as outfile2:
			writer2 = csv.writer(outfile2)
			writer2.writerow(["Ent", "Label_"])

			for line in text:
				#print (line)
				#line = [x.decode('utf8') for x in line]
				line = re.sub(r'[^a-zA-Z0-9]+',' ',line)
				line = re.sub('\.', '', line)
				#exclude = set(string.punctuation)
				#line = ''.join(ch for ch in line if ch not in exclude)
				print (line)
				doc = nlp(line)
				print (doc)
				for token in doc:
					print(token)
					writer1.writerow([token, token.lemma_, token.pos_])
					#print (token, token.lemma, token.lemma_, token.pos, token.pos_)
				for ent in doc.ents:	
					print (ent, ent.label, ent.label_)
					writer2.writerow([ent, ent.label_])

def extract_entities_spacy_2(line):
	#print (line)
	
	line = re.sub(r'[^a-zA-Z0-9]+',' ',line)
	line = re.sub('\.', '', line)
	doc = nlp(line)
	return doc

def extract_NNP(entities):
 	NNP_list =  [s for s in tagged if s[1] == 'NNP']
 	return (NNP_list)

def extract_tags(text):
	tokens=nltk.word_tokenize(text)
	tagged=nltk.pos_tag(tokens)
	return (tagged)

def extract_entities(tagged):
	entities = ne_chunk(tagged)
	return (entities)

def read_data():
	train = pd.read_csv("171114_Wildcat_Full_dedupe_text_split.csv", header=0, quoting=1,encoding = 'latin_1') 
	#train = pd.read_csv("171114_Wildcat_Full_dedupe_text.csv")
	#train = pd.read_csv("export.csv")
	train.shape
	train.columns.values
	train['Question3abc'] = train[train.columns[1:4]].apply(lambda x: ','.join(x.dropna()),axis=1)
	#train = train.drop_duplicates(subset=['Question3abc'], keep=False)
	return (train['Question3abc'])

def stanford_tag(line):

	java_path ='C:\Program Files\Java\jdk1.8.0_131\jre.exe'
	nltk.internals.config_java(java_path)
	# NERTagger
	stanford_dir = 'C:\Python\Python36\Lib\stanford-ner-2017-06-09'
	jarfile = stanford_dir + '\stanford-ner.jar'
	modelfile = stanford_dir + '\classifiers\english.conll.4class.distsim.crf.ser.gz'

	st = StanfordNERTagger(model_filename=modelfile, path_to_jar=jarfile)
	st = StanfordNERTagger(modelfile, stanford_dir +'/stanford-ner.jar')
	tagged_st=st.tag(line)
	#for tag in tagged_st:
	#	if tag[1] in ["PERSON", "LOCATION", "ORGANIZATION"]: print (tag)
	for tag, chunk in groupby(tagged_st, lambda x:x[1]):
		if tag != "O": print ("%-12s"%tag, " ".join(w for w, t in chunk))

def extract_entities_nltk(text):

	with open('Entities_nltk.csv', 'w', newline='') as outfile:
		writer = csv.writer(outfile)
		writer.writerow(["Entity", "Word", "pos_tag"])
		for line in text:
			line = re.sub(r'[^\w\s]','',line)
			
			for sentence in sent_tokenize(line):
			    chunks = ne_chunk(pos_tag(word_tokenize(sentence)))
			    entities.extend([chunk for chunk in chunks if hasattr(chunk, 'label')])
			    for j in entities:
			    	print(j.label(),j[0][0],j[0][1])
			    	writer.writerow([j.label(),j[0][0],j[0][1]])

def extract_entities_nltk_2(text):
	for line in text:
		#print(line)
		line = re.sub(r'[^\w\s]','',line)
		for sentence in sent_tokenize(line):
			chunks = ne_chunk(pos_tag(word_tokenize(sentence)))
			entities_nltk.extend([chunk for chunk in chunks if hasattr(chunk, 'label')])
			#for j in entities_nltk:
			#    print(j.label(),j[0][0],j[0][1])

def read_data_2(filename):
	text = open(filename,'r', encoding = "ISO-8859-1") 
	#encoding="utf8
	return text 
    	

if __name__ == '__main__':
	nlp = spacy.load('en_core_web_sm')
	#Add all the files to be scanned in the fileList
	entities_nltk = []
	entities_spacy= []
	tokens_spacy=[]
	temp={}
	#os.chdir(".\\DATA\\EmotionIntensity\\Train\\")
	#fileList=glob.glob("*.txt")
	fileList=[]
	fileList.append("C:\\Python\\Python36\\DATA\\171114_Wildcat_Full_dedupe_text.csv")
	print(fileList)

	
	#The following code block extra entities using NLTK package and write to file in .\OUTPUT\Entities_nltk.csv
	with open('.\\Output\\Entities_nltk.csv', 'w', newline='',encoding = 'utf-8') as outfile:
		print ("Processing with NLTK now.....")
		writer = csv.writer(outfile)
		writer.writerow(["Entity", "Word", "pos_tag"])
		for fl in fileList:
			text=read_data_2(fl) #modify the read data function such that it has all the lines of text in it
			extract_entities_nltk_2(text)
		print (entities_nltk)
		for e in entities_nltk:
			writer.writerow([e.label().encode("utf-8"),e[0][0],e[0][1]])
			#print([e.label().encode("utf-8"),e[0][0],e[0][1]])
	
	#The following code block extra entities using Spacy package and write to file in .\OUTPUT\POStags_Lemma_Spacy.csv and .OUTPUT\Entities_Spacy.csv
	with open('.\\Output\\POStags_Lemma_Spacy.csv', 'w', newline='',encoding = 'utf-8') as outfile1:
		print ("Processing with Spacy Now.....")
		writer1 = csv.writer(outfile1)
		writer1.writerow(["Token", "Token_lemma_", "pos_tag (Spacy)"])
		with open('.\\Output\\Entities_Spacy.csv', 'w', newline='') as outfile2:
			writer2 = csv.writer(outfile2)
			writer2.writerow(["Ent", "Label_"])
			for fl in fileList:
				text=read_data_2(fl)
				for t in text:
					#print(t)
					doc = extract_entities_spacy_2(t)
					#print (doc)
					for token in doc:
						#print(token)
						tokens_spacy.extend([token, token.lemma_, token.pos_])
						writer1.writerow([token, token.lemma_, token.pos_])
					for ent in doc.ents:	
						#print (ent, ent.label, ent.label_)
						entities_spacy.extend([ent, ent.label_])
						writer2.writerow([ent, ent.label_])
	print ("Check Output files")