# coding: utf-8
# 2分类机器学习模型，方法：SVM，LR，NB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, metrics
import pickle
import time
import codecs
from sklearn2pmml import PMMLPipeline
from sklearn.externals import joblib

start = time.clock()

class WAF(object):
    def __init__(self):
        print('读取语料库：')
        seed_list, content_list = self.get_data('./data/豆瓣')  # 文件格式：老无所依\t差评\t我不能因为它得了奥斯卡就说明它好看，我不能因。。。

        print('\t'+'好评数:' + str(len(seed_list)) + '  差评数:' + str(len(content_list)))

        seed_y = [0 for i in range(0, len(seed_list))]
        content_y = [1 for i in range(0, len(content_list))]

        queries = content_list + seed_list
        y = content_y + seed_y

        # 数据矢量化
        self.vectorizer = TfidfVectorizer(tokenizer=self.get_ngrams)
        X = self.vectorizer.fit_transform(queries)
        print('向量化后维度：' + str(X.shape))

        print('划分训练集、测试集...')
        # 使用 train_test_split 分割 X y 列表
        # X_train矩阵的数目对应 y_train列表的数目(一一对应)  -->> 用来训练模型
        # X_test矩阵的数目对应 	 (一一对应) -->> 用来测试模型的准确性
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)
        print('划分完成，训练集开始训练分类器...')

        #self.model = LogisticRegression()
        self.model = svm.SVC()
        #self.model=MultinomialNB(alpha=0.001)
        self.pipeline = PMMLPipeline([("classifier",self.model)])
        self.pipeline.fit(X_train, y_train)

        joblib.dump(self.pipeline, "./result/classifier.pkl.z", compress=9)  # compress压缩程度

        print('训练完毕!!!  测试集开始预测结果...')
        predict = self.pipeline.predict(X_test)
        print("精度:{0:f}".format(metrics.precision_score(y_test, predict, average="weighted")))
        print("召回:{0:f}".format(metrics.recall_score(y_test, predict, average="weighted")))
        print("f1-score:{0:f}".format(metrics.f1_score(y_test, predict, average="weighted")))
        print("预测完毕！！！！")
        print('***********************************************************')
        print('***********************************************************')

    def predict(self):
        new_pos,new_neg = self.get_data('./data/影评')   # 格式与训练数据一样
        new=new_pos + new_neg
        pos_y = [0 for i in range(0, len(new_pos))]
        neg_y = [1 for j in range(0, len(new_neg))]
        new_y = pos_y + neg_y
        X_predict = self.vectorizer.transform(new)
        print('新数据向量化后维度：' + str(X_predict.shape))
        res = self.model.predict(X_predict)
        print("精度:{0:f}".format(metrics.precision_score(new_y, res, average="weighted")))
        print("召回:{0:f}".format(metrics.recall_score(new_y, res, average="weighted")))
        print("f1-score:{0:f}".format(metrics.f1_score(new_y, res, average="weighted")))
        print("预测完毕！！！！")

    def get_data(self,path):
        f = open(path, 'r', encoding='utf8')
        pos = []
        neg = []
        for line in f.readlines():
            text = line.strip().split('\t')
            if len(text) == 3 and text[1] == '好评':
                pos.append(text[2])
            if len(text) == 3 and text[1] == '差评':
                neg.append(text[2])
        return pos, neg

    def get_ngrams(self, query):
        tempQuery = str(query)
        ngrams = []
        for i in range(0, len(tempQuery) - 2):
            ngrams.append(tempQuery[i:i + 2])
        return ngrams

if __name__ == '__main__':
    w = WAF()
    # 检测模型文件lgs.pickle不存在,若不在，需先训练出模型
    with open('./result/classifier.pickle', 'wb') as output:
        pickle.dump(w, output)
    with open('./result/classifier.pickle', 'rb') as input:
        w = pickle.load(input)
    w.predict()

end = time.clock()
print('\n')
print('=========' + '耗时%ds' % (end - start)+'=========')
