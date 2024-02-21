import bs4 as bs
import urllib.request
import re
import nltk
import sys, getopt
import collections
import heapq

# Number of vendor sentences to evaluate
num_sentences_to_evaluate = 5


# Max length of sentences to evaluate
max_words_in_sentence = 50


## From https://stackabuse.com/text-summarization-with-nltk-in-python/
## NLTK install instructions are at https://www.nltk.org/

# Read the first argument file
filename = sys.argv[1]
text_file = open(sys.argv[1], "r")
text = text_file.read()

article_text = re.sub(r'\[[0-9]*\]', ' ', text)
article_text = re.sub(r'\s+', ' ', article_text)


# Removing special characters and digits
formatted_article_text = re.sub('[^a-zA-Z0-9\.\,]', ' ', article_text )
#formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)


sentence_list = nltk.sent_tokenize(article_text)

stopwords = nltk.corpus.stopwords.words('english')


# Count word frequencies on lines
word_frequencies = {}
for word in nltk.word_tokenize(formatted_article_text):
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

# Build relative frequencies by dividing all  word frequencies with the highest frequency of a word.
maximum_frequency = max(word_frequencies.values())

for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/maximum_frequency)


# Build sums of relative frequencies for every line
sentence_scores = {}
for sent in sentence_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < max_words_in_sentence:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]


i = 0
curr_score = 0
vendor_name = ""



# Store vendor's total sentence scores in a dictionary
vendor_scores = {}

vendor_name_pattern = "(---.*---)"

# Find vendor names that start and end with ---
for sent in sentence_scores:
    currline = re.sub('[^a-zA-Z0-9\.\,\-\&]', '', sent)
    
    # We found a new vendor name
    if re.match(vendor_name_pattern, currline):
        vendor_name = re.match(vendor_name_pattern, currline).group(1)
        # print ("New Vendor", vendor_name)
    # Add frequencies for the next 5 lines
    curr_score = curr_score + sentence_scores[sent]
    i = i + 1
    # Assume that we 
    if i == num_sentences_to_evaluate:
        # print (vendor_name, "score", curr_score)
        vendor_scores[vendor_name] = curr_score
        i = 0
        curr_score = 0
        
# Sort vendor scores x[1] means that it's the second column in the dictionary, this creates a list, not a dictionary
sorted_vendor_scores = sorted(vendor_scores.items(), key=lambda x: x[1], reverse=True)


# Print out sentence scores in detail
print ("Detailed sentence scores")
for sent in sentence_scores:
    print(sent, sentence_scores[sent])

print ("Vendor's total scores")
# Print sorted vendor scores out for import into a CSV file
for i in sorted_vendor_scores:
       print (re.sub('-', '', i[0]), ",", i[1])

#  End program

## Input format, remove leading #s

#---BAE---
#3rd party integration: NR automatically visualises multi-layer corporate
#structures sourced from 3rd party data eg Dun & Bradstreet in investigation.
#Portfolio-wide Alert prioritisation: Intelligent Event Triage (IET)  functionality uses alert enrichment & ML to prioritise alerts in order of urgency, so investigators time is spent on the most urgent alerts first.
#Name matching algorithms, Allowing customisation of standard algorithms (Build Your Own Algorithm) & associating lists of words (nicknames, financial terms) to detect more variance and focus on what is important (Synonyms and weighted words).
#Python integration and expansion: Adding to key models (eg XGBoost and UMAP) with inclusion of two further classification models Histogram Gradient Boosting & Multi-Layer Perceptron, further commitment to open source data science and offer modelling flexibility. 
#24/7 architecture: Architecture separates processing from the user interfaces for maximum resilience, scalability, ease of upgrade and performance optimization.
#
#---Azentio---
#Support for AMLOCK on PostgreSQL DB ( Currently it is supported on Oracle) This would reduce TCO for smaller customers. 
#Smart Card Reader Service for Entity KYC Validations & DOPA Authentication with Thai Regulatory for all entities. 
#Transliteration Service from English to Arabic & vice-versa.
#Webservices / MQ Integration with Enterprise legacy for asynchronous / synchronous communication.
#AI/ ML Models for Analytics for False Positive reduction, Alert scoring.
#
#---Feedzai---
#Patent-pending models increase model fairness by 93%, Developed FATE to be used with fairness metrics, model metrics, and sensitive PII, It works with any algorithm and model settings.
#Alert Prioritization, Both entity-centric features and attributes characterizing inter-entity relations in the form of graph-based features, Models validated on real-world data show reduction in false positives by 80% while detecting over 90% of true positives. 
#Automation of data migration and run multiple threshold tuning exercises in parallel to reduce time to value.
#New Payment Screening solution provides continuous monitoring and evaluation of global watchlists with quick adoption to new sanctions.
#New KYC/CDD solution continuously risk scores an FIs customer base without manual effort, As the customers interact with the FI their risk score is consistently updated, Alerts are only generated when significant change in risk.
