## From https://www.numpyninja.com/post/text-summarization-through-use-of-spacy-library
import spacy

import sys, getopt

from spacy.lang.en.stop_words import STOP_WORDS

stopwords=list(STOP_WORDS)
from string import punctuation
punctuation=punctuation+ '\n'


text_file = open(sys.argv[1], "r")
text = text_file.read()

nlp = spacy.load('en_core_web_sm')
doc= nlp(text)
tokens=[token.text for token in doc]
#print(tokens)

word_frequencies={}
for word in doc:
    if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1



#print(word_frequencies)

max_frequency=max(word_frequencies.values())
for word in word_frequencies.keys():
    word_frequencies[word]=word_frequencies[word]/max_frequency

#print(word_frequencies)

sentence_tokens= [sent for sent in doc.sents]
#print(sentence_tokens)

sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():                            
             sentence_scores[sent]=word_frequencies[word.text.lower()]
            else:
             sentence_scores[sent]+=word_frequencies[word.text.lower()]

sentence_scores

from heapq import nlargest
# select_length=int(len(sentence_tokens)*0.1)
select_length=3
# select_length
summary=nlargest(select_length+1, sentence_scores,key=sentence_scores.get)

#summary

final_summary=[word.text for word in summary]
#final_summary
summary=''.join(final_summary)
print(summary)

