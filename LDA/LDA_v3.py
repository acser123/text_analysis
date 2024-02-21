import gensim
import codecs
class MySentences(object):
    def __init__(self,filename):
        self.filename=filename
    def __iter__(self):
        with codecs.open(self.filename) as f:
            for line in f.readlines():
                wordlist=list()
                for word in line:
                    wordlist.append(word)
                yield wordlist

sentences=MySentences('C:\\Users\\acser\\Documents\\Python\\text_summary\\Q1_ExecRoadmap.txt')
model=gensim.models.Word2Vec(sentences)
model.save('w.model')