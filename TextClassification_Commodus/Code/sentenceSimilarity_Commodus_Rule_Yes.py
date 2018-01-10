from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import os
import glob
import csv
import time

##http://nlpforhackers.io/wordnet-sentence-similarity/

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
    

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score, count = 0.0, 0
 
    # For each word in the first sentence
    for synset in synsets1:

        temp = [synset.path_similarity(ss) for ss in synsets2]

    
        #print("Adding None now")
        temp = [0.0 if v is None else v for v in temp]
  
        #print(temp)
        # Get the similarity value of the most similar word in the other sentence
        #print(len(temp))

        if (len(temp)) :
            best_score = max(temp)
        else:
            best_score = 0.0
            count=1
        # Check that the similarity could have been computed
        if best_score is not 0.0:
            score += best_score
            count += 1
 
    # Average the values
    score /= count
    return score


focus_sentence = "do you confirm you are over 18, authorised to enter into this agreement, and understand and accept the terms and conditions of this offer, these are also on Powershop’s website.  Please respond with yes or no"
rule = "Rule_Yes"
flag=0

os.chdir("G:\\NLP\\Commodus\\Test\\Test_small\\")
fileList=glob.glob("*.txt")

with open("G:\Git_code\AudioAnalysis\TextClassification_Commodus\Output\Test.csv", 'w', newline='',encoding = 'utf-8') as outfile1:
    writer1 = csv.writer(outfile1)
    writer1.writerow(["ID","Rule","Sentence", "Next Sentence","Score"])

    for fl in fileList:
        print(fl)
        test = open(fl)
        results=[]
        for line in test:
            sentences = line.split(".")
            for sentence in sentences:
                if sentence.strip():
                    #print(flag, sentence)
                    #if (flag == 1 and (sentence == "Yes" or "Yeah" or "Yup")):
                    if (flag == 1 and (sentence == "Yes" or "Yeah" or "Yup")):
                        print (previous, " AND next sentence --> ", sentence)
                        #writer1.writerow([fl,rule,previous,sentence,score])
                        #time.sleep(10)
                        flag=0
                    score=(sentence_similarity(focus_sentence, sentence))
                    if(score>0.5 and rule=="Rule_Yes"):
                        #print ("Similarity(\"%s\", \"%s\") = %s in Document : %s" % (focus_sentence, sentence, sentence_similarity(focus_sentence, sentence),fl))
                        previous = sentence
                        flag =1
                        results.append(score)        #print(max(results))            
 

   