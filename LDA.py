#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import jieba
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import os
import codecs
import logging
from lxml import etree
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


train = []
# huge_tree=True, 防止文件过大时出错 XMLSyntaxError: internal error: Huge input lookup
parser = etree.XMLParser(encoding='utf8',huge_tree=True)
def load_data(dirname):
    global train
    files = os.listdir(dirname)
    for fi in files:
        logging.info("deal with "+fi)
        text = codecs.open(dirname+fi, 'r',encoding='utf-8',errors='ignore').read()
        # xml自身问题，存在&符号容易报错, 用&amp;代替
        text = text.replace('&', '&amp;')
        # 解析xml，提取新闻标题及内容
        root = etree.fromstring(text, parser=parser)
        docs = root.findall('doc')
        for doc in docs:
            tmp = ""
            for chi in doc.getchildren():
                if chi.tag == "contenttitle" or chi.tag == "content":
                    if chi.text != None and chi.text != "":
                        tmp += chi.text
            if tmp != "":
                train.append(tmp)


if __name__=="__main__":
    dirname="E:/data/SogouNews/111/"
    load_data(dirname)

#语料库分词、去停止词

#stopwords = codecs.open('e:/data/stopwords/stopwords.txt','r',encoding='utf-8',errors='ignore').readlines()
stopwords = [line.strip() for line in open('e:/data/stopwords/stopwords.txt').readlines()]
#stopwords = [ w.strip() for w in stopwords ]
train_set = []
for line in train:
    line = list(jieba.cut(line))
    train_set.append([ w for w in line if w not in stopwords ])
    #line=jieba.cut(line)
    #train_set.append(' '.join(list(set(train) - set(stopwords))))  内存不够

#构建训练语料库
dictionary = Dictionary(train_set)
#dictionary.filter_n_most_frequent(2)  #过滤频率最高的2个词
#dictionary.filter_extremes(no_below=5, no_above=0.7, keep_n=100)  #去高低频词
corpus = [ dictionary.doc2bow(text) for text in train_set]


# lda模型训练
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=6)
lda.print_topics(6)





