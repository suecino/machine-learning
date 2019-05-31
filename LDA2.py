#LDA
import jieba
from gensim import corpora
from gensim import models
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary

fr = open('e:/data/鹰眼/phrase.txt', 'r',encoding='gbk')
train = []
for line in fr.readlines():
    line = line.split(' ')
    train.append(line)

print(len(train))
print(' '.join(train[2]))

dictionary = corpora.Dictionary(train)
corpus = [dictionary.doc2bow(text) for text in train]
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)

topic_list = lda.print_topics(10)
#print(type(lda.print_topics(20)))
print(len(lda.print_topics(10)))

for topic in topic_list:
    print(topic)
