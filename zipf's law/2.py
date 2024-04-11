
from nltk import FreqDist
import os
import matplotlib
import matplotlib.pyplot as plt
import jieba
 
#返回目录下文件所有的绝对路径，不管是子目录还是孙目录，
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(os.path.join(root, file))
    return L
 
#得到所有的词语，以字典的方式存储起来
def getChineseTerms(files_path):
    res = {}
    for i in range(len(files_path)):
        filename = files_path[i]
        with open(filename, 'rb') as f:
            mytext = f.read()#.decode('utf-8')
            mytext = " ".join(jieba.cut(mytext))#得到具体的词语
            # print(mytext)
            myword = [i for i in mytext.strip().split() if len(i) >= 2]  # split默认将所有的空格删除
            # print(myword)
            for j in myword:
                res[j] = res.get(j, 0) + 1
    return res
 
#画图显示
def showPlt(res):
    # res1 = sorted(res.items(), key=lambda x: x[1], reverse=True)  # 不是在原来上修改
    # print(res1)
    ranks = []
    freqs = []
    for rank, value in enumerate(res1):  # 0 ('的', 87343)
        ranks.append(rank + 1)
        freqs.append(value[1])
        rank += 1
    plt.loglog(ranks, freqs)
    plt.xlabel('词语频数', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.ylabel('词语名次', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.grid(True)
    plt.show()
 
if __name__ == '__main__':
 
    files_path = file_name(r'C:\Users\qizh1\Desktop\深度学习与nlp\中文语料库')
    res = getChineseTerms(files_path)
    res1 = sorted(res.items(), key=lambda x: x[1], reverse=True)
    print(res1)
    showPlt(res1)