#!--encoding=utf-8


from __future__ import print_function

import logging
import os
import re
from collections import defaultdict
from time import time
from six.moves import range
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)



def loadDataset():
    '''导入文本数据集'''
    Dataset=[]
    for line in open('e:/data/鹰眼/原数据/yn500.txt', 'r',encoding='utf-8').readlines():
        #print(line)
        Dataset.append(line.strip())
    return Dataset


t0 = time()
logger.info('TfidfVectorizer...')
dataset=loadDataset()
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, use_idf=True)
X = vectorizer.fit_transform(dataset)
word=vectorizer.get_feature_names()
#f=open('e:/data/鹰眼/word.txt','w',encoding='utf-8')
#f.write(str(word))
logger.info('vectorizer: %fs' % (time() - t0))

t0 = time()
logger.info('MiniBatchKMeans...')
km = MiniBatchKMeans(n_clusters=10, batch_size=1000)
km.fit(X)
logger.info('kmeans: %fs' % (time() - t0))

t0 = time()
logger.info('collecting result')
pred_labels = km.labels_
result = defaultdict(list)
for idx in range(len(pred_labels)):
    result[pred_labels[idx]].append(dataset[idx])
for k in result:
    name = 'res-{:d}.txt'.format(k)
    elems = result[k]
    out_path = os.path.join('e:/data/鹰眼/cluster1', name)
    with open(out_path, 'w',encoding='utf-8') as fout:
        logger.info('writing %s' % out_path)
        for elem in elems:
            fout.write(elem)
            fout.write('\n')
logger.info('finished: %fs' % (time() - t0))
sorted_indices = km.cluster_centers_.argsort()[:, ::-1]
id2words = vectorizer.get_feature_names()
for i in range(km.n_clusters):
    print('cluster: %i' % i)
    for idx in sorted_indices[i, :20]: #输出每一类的得分最高的特征的个数
        print(' %s' % id2words[idx])
    print()



