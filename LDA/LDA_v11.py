## From https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/

if __name__ == '__main__':

   doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
   doc2 = "My father spends a lot of time driving my sister around to dance practice."
   doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
   doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
   doc5 = "Health experts say that Sugar is not good for your lifestyle."

   # compile documents
   doc_complete = [doc1, doc2, doc3, doc4, doc5]


   import bs4 as bs
   import urllib.request
   import re
   import nltk

   impfile = ""

   number_of_sentences_in_summary = 7
   max_words_in_sentence = 20


   import sys, getopt

   ## From https://stackabuse.com/text-summarization-with-nltk-in-python/
   ## NLTK install instructions are at https://www.nltk.org/

   number_of_topics = int(sys.argv[1])
   number_of_words_in_topic = int(sys.argv[2])
   # scraped_data = urllib.request.urlopen(sys.argv[3])
   # scraped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')
   # article = scraped_data.read()

   # parsed_article = bs.BeautifulSoup(article,'lxml')

   # paragraphs = parsed_article.find_all('p')
   # print (parsed_article)

    


   ###
   file = open(sys.argv[3], 'r', encoding="utf8")
   article_text = file.readlines()

   # print (article_text)

   doc_complete = article_text

   from nltk.corpus import stopwords
   from nltk.stem.wordnet import WordNetLemmatizer
   import string
   stop = set(stopwords.words('english'))
   exclude = set(string.punctuation)
   lemma = WordNetLemmatizer()
   def clean(doc):
       stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
       punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
       normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
       return normalized

   doc_clean = [clean(doc).split() for doc in doc_complete]  

   # Importing Gensim
   import gensim
   from gensim import corpora
   from gensim.test.utils import common_corpus, common_dictionary

   # Creating the term dictionary of our courpus, where every unique term is assigned an index. 
   dictionary = corpora.Dictionary(doc_clean)

   # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
   doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

   # Creating the object for LDA model using gensim library
   # Lda = gensim.models.ldamodel.LdaModel
   Lda = gensim.models.LdaMulticore

   # Running and Trainign LDA model on the document term matrix.
   ldamodel = Lda(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary, passes=100)

   print(ldamodel.print_topics(num_topics=number_of_topics, num_words=number_of_words_in_topic))

