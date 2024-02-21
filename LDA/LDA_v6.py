import gensim
import os
from gensim.utils import simple_preprocess

from gensim.test.utils import common_corpus, common_dictionary

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

#processed_docs = preprocess(newsgroups_train)

#dictionary = gensim.corpora.Dictionary(processed_docs)

#bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


lda_model =  gensim.models.LdaMulticore(common_corpus, 
                                   num_topics = 8, 
                                   id2word = common_dictionary,                                    
                                   passes = 10,
                                   workers = 2)


#lda_model =  gensim.models.LdaMulticore(bow_corpus, 
#                                   num_topics = 8, 
#                                   id2word = dictionary,                                    
#                                   passes = 10,
#                                   workers = 2)

