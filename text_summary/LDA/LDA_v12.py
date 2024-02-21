## From https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
### the if __name__ == '__main__' is required for multithreaded LDA model execution
if __name__ == '__main__':


   from nltk.corpus import stopwords
   from nltk.stem.wordnet import WordNetLemmatizer

   import string
   import sys, getopt

   # Importing Gensim
   import gensim
   from gensim import corpora
   from gensim.test.utils import common_corpus, common_dictionary

   number_of_LDA_model_passes = 50
   number_of_topics = int(sys.argv[1])
   number_of_words_in_topic = int(sys.argv[2])
  
   ### Read the input file into a list, line by line
   file = open(sys.argv[3], 'r', encoding="utf8")
   doc_complete = file.readlines()

   # Prepare for clean up the document list
   stop = set(stopwords.words('english'))
   exclude = set(string.punctuation)
   lemma = WordNetLemmatizer()
   def clean(doc):
       stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
       punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
       normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
       return normalized

   # Clean up the document list
   doc_clean = [clean(doc).split() for doc in doc_complete]  

   # Creating the term dictionary of our courpus, where every unique term is assigned an index. 
   dictionary = corpora.Dictionary(doc_clean)
   

   # print ([dictionary[doc] for doc in dictionary])

   # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
   doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

   # Creating the object for LDA model using gensim library
   # Lda = gensim.models.ldamodel.LdaModel
   Lda = gensim.models.LdaMulticore

   # Running and Trainign LDA model on the document term matrix. Remove the random_state=0 if randomization is required.
   ldamodel = Lda(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary, passes=number_of_LDA_model_passes, random_state=0)

   print(ldamodel.print_topics(num_topics=number_of_topics, num_words=number_of_words_in_topic))

