#!/usr/bin/env python
# _*_ coding:utf-8 _*_

# 多分类

import jieba
import os

#读取原始语料库
def savefile(savepath,content):
    with open(savepath,"w",encoding='gb2312',errors='ignore') as fp:
        fp.write(content)

def readfile(path):
    with open(path,"r",encoding='gb2312',errors='ignore') as fp:
        content = fp.read()
    return content

#给语料库一份一份地分词、去停止词
def corpus_seg(corpus_path,seg_path):
    catelist=os.listdir(corpus_path)
    for mydir in catelist:
        class_path=corpus_path+mydir+"/"
        seg_dir=seg_path+mydir+"/"

        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)
        stop = [line.strip() for line in open('e:/data/stopwords/stopwords.txt').readlines()]
        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = class_path+file_path
            content = readfile(fullname)

            content=content.replace("\r\n", "")
            content=content.replace(" ", "")
            content_seg=jieba.cut(content)
            #savefile(seg_dir+file_path," ".join(content_seg))  #未去停止词
            savefile(seg_dir + file_path, " ".join(list(set(content_seg)-set(stop))))  #去停止词
    print("中文语料库分词结束！！！")


if __name__ == "__main__":
    corpus_path = "E:/data/TanCorpMinTrain/"
    seg_path = "E:/ndata/train_seg/"
    corpus_seg(corpus_path, seg_path)

    corpus_path = "E:/data/TanCorpMinTest/"
    seg_path = "E:/ndata/test_seg/"
    corpus_seg(corpus_path, seg_path)

# 文本文件转换成Bunch类型
import os
import pickle
from sklearn.datasets.base import Bunch


def corpus2Bunch(woedbag_path,seg_path):
    catelist = os.listdir(seg_path)
    bunch=Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)

    for mydir in catelist:
        class_path=seg_path+mydir+"/"
        file_list=os.listdir(class_path)
        for file_path in file_list:
            fullname=class_path+file_path
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(readfile(fullname))

    with open(wordbag_path,"wb") as file_obj:
            pickle.dump(bunch,file_obj)
    print("构建文本对象结束！！！")


if __name__=="__main__":
    wordbag_path="e:/ndata/train_word_bag/train_set.dat"
    seg_path="e:/ndata/train_seg/"
    corpus2Bunch(wordbag_path,seg_path)

    wordbag_path="E:/ndata/test_word_bag/test_set.dat"
    seg_path="e:/ndata/test_seg/"
    corpus2Bunch(wordbag_path,seg_path)


#构建if-idf词向量空间
from sklearn.feature_extraction.text import TfidfVectorizer

def _readfile(path):
    with open(path,"r",encoding='gb2312', errors='ignore') as fp:
        content=fp.read()
    return content

def _readbunchobj(path):
    with open(path,"rb")as file_obj:
        bunch=pickle.load(file_obj)
    return bunch

def _writebunchobj(path,bunchobj):
    with open(path,"wb") as file_obj:
        pickle.dump(bunchobj,file_obj)

def vector_space(stopword_path,bunch_path,space_path,train_tfidf_path=None):
    stpwdlst=_readfile(stopword_path).splitlines()
    bunch=_readbunchobj(bunch_path)
    tfidfspace=Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})
    vectorizer=TfidfVectorizer(stop_words=stpwdlst,sublinear_tf=True,max_df=0.5)

    if train_tfidf_path is not None:
        trainbunch=_readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary=trainbunch.vocabulary
        vectorizer=TfidfVectorizer(stop_words=stpwdlst,sublinear_tf=True,max_df=0.5,vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm=vectorizer.fit_transform(bunch.contents)
    else:
        vectorizer=TfidfVectorizer(stop_words=stpwdlst, sublinear_tf=True, max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_

    _writebunchobj(space_path, tfidfspace)
    print("if-idf词向量空间实例创建成功！！！")

if __name__=="__main__":
    stopword_path="e:/data/stopwords/stopwords.txt"
    bunch_path="e:/ndata/train_word_bag/train_set.dat"
    space_path="e:/ndata/train_word_bag/tfdifspace.dat"
    vector_space(stopword_path, bunch_path, space_path)

    bunch_path = "e:/ndata/test_word_bag/test_set.dat"
    space_path = "e:/ndata/test_word_bag/testspace.dat"
    train_tfidf_path= "e:/ndata/train_word_bag/tfdifspace.dat"
    vector_space(stopword_path, bunch_path, space_path,train_tfidf_path)


#构建贝叶斯分类器

from sklearn.naive_bayes import MultinomialNB   # 导入多项式分布的贝叶斯算法

def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
        return bunch


#导入训练集
trainpath="e:/ndata/train_word_bag/tfdifspace.dat"
train_set=_readbunchobj(trainpath)

#导入测试集
testpath="e:/ndata/test_word_bag/testspace.dat"
test_set=_readbunchobj(testpath)

#训练分类器：输入词袋向量和分类标签  alpha越小 迭代次数越多，精度越高
clf=MultinomialNB(alpha=0.001).fit(train_set.tdm,train_set.label)

# 预测分类结果
predicted = clf.predict(test_set.tdm)
for flabel, file_name, expect_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel != expect_cate:
        print (file_name, ":实际类别：", flabel, "-->预测类别：", expect_cate)
print("预测完毕！！！")

# 计算分类精度

from sklearn import metrics
def metrics_result(actual,predict):
    print ("精度:{0:.3f}".format(metrics.precision_score(actual,predict,average="weighted")))
    print ("召回:{0:.3f}".format(metrics.recall_score(actual,predict,average="weighted")))
    print ("f1-score:{0:.3f}".format(metrics.f1_score(actual,predict,average="weighted")))

metrics_result(test_set.label,predicted)

















