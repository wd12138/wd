from nltk import FreqDist
import os
import matplotlib
import matplotlib.pyplot as plt
 
 
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(os.path.join(root, file))
    return L
 
 
def getCharacter(files_path):
    res = {}
    for i in range(1, len(files_path)):
        path = files_path[i]
        for x in open(path, 'rb').read().decode('utf-8',errors='ignore'):
            if 19968 <= ord(x) <= 40869:
                res[x] = res.get(x, 0) + 1
    return res
 
def showPlt(res):
    res1 = sorted(res.items(), key=lambda x: x[1], reverse=True)  
    ranks = []
    freqs = []
    for rank, value in enumerate(res1):  # 0 ('的', 87343)
        ranks.append(rank + 1)
        freqs.append(value[1])
        rank += 1
    plt.loglog(ranks, freqs)
    plt.xlabel('汉字频数', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.ylabel('汉字名次', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.grid(True)
    plt.show()
files_path = file_name(r'C:\Users\qizh1\Desktop\深度学习与nlp\中文语料库')
res = getCharacter(files_path)
showPlt(res)

for key, value in res.items():
    encoded_key = key.encode('utf-8')
    encoded_value = value
    print(encoded_key.decode('utf-8'), ':', encoded_value)