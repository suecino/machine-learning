# 1 FP-Growth发现共现词
import fp_growth as fpg

def getline(thefilepath, line_num):
    if line_num < 1: return ''
    for currline, line in enumerate(open(thefilepath, 'rU')):
        if currline == line_num - 1: return line
    return ''

'''
例子
lists = [
    ['啤酒', '牛奶', '可乐'],
    ['尿不湿', '啤酒', '牛奶', '橙汁'],
    ['啤酒', '尿不湿'],
    ['啤酒', '可乐', '尿不湿'],
    ['啤酒', '牛奶', '可乐']
]
print(lists)   
'''

if __name__ == '__main__':
    #调用find_frequent_itemsets()生成频繁项
    #@:param minimum_support表示设置的最小支持度，即若支持度大于等于inimum_support，保存此频繁项，否则删除
    #@:param include_support表示返回结果是否包含支持度，若include_support=True，返回结果中包含itemset和support，否则只返回itemset

    # 生成多维数组
    ff = open('./data/title_seg.txt', 'r')
    num = len(ff.readlines())
    lists = []
    for i in range(1, num):
        lists.append(getline(r'./data/title_seg.txt', i).strip().split(' '))
    print(lists)

    frequent_itemsets = fpg.find_frequent_itemsets(lists, minimum_support=4, include_support=True)

    result = []
    for itemset, support in frequent_itemsets:
        result.append((itemset, support))

    c={}
    fw=open('./result/fp_growth.txt','w')
    result = sorted(result, key=lambda i: i[0])   # 排序后输出
    for itemset, support in result:
        if support>=28 and len(''.join(itemset).strip())<=15 and len(''.join(itemset).strip())>=4:
            print(str(itemset) + ' '   + str(support))
            c[support]=''.join(itemset).strip()
            fw.write(''.join(itemset).strip()+'\t'+str(support)+'\n')
