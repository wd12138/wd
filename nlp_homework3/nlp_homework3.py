import os
import re
import jieba
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def read_novel(path_in, path_out):
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    names = os.listdir(path_in)
    for name in names:
        novel_name = os.path.join(path_in, name)
        fenci_name = os.path.join(path_out, name)
        try:
            with open(novel_name, 'r', encoding='ANSI') as file, open(fenci_name, 'w', encoding='utf-8') as f_out:
                for line in file:
                    line = line.strip()
                    line = re.sub("[A-Za-z0-9\：\·\—\，\。\“\”\\n \《\》\！\？\、\...]", "", line)
                    line = content_deal(line)
                    con = jieba.cut(line, cut_all=False)
                    f_out.write(" ".join(con) + "\n")
            print(f"Processed and saved: {fenci_name}")
        except Exception as e:
            print(f"Error processing {novel_name}: {e}")
    return names

def content_deal(content):
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b', '她', '他', '你', '我', '它', '这']
    for a in ad:
        content = content.replace(a, '')
    return content

if __name__ == '__main__':
    input_path = r'C:\Users\qizh1\Desktop\jyxstxtqj_downcc.com'
    output_path = r'C:\Users\qizh1\Desktop\output'
    
    files = read_novel(input_path, output_path)

    test_names = ['张无忌', '乔峰', '郭靖', '杨过', '令狐冲', '韦小宝']
    test_menpai = ['明教', '逍遥派', '少林', '全真教', '华山派', '少林']
    
    for i in range(min(len(files), len(test_names))):
        name = os.path.join(output_path, files[i])
        if not os.path.isfile(name):
            print(f"File not found: {name}")
            continue

        print(f"Training model on {name}...")
        try:
            model = Word2Vec(sentences=LineSentence(name), hs=1, min_count=10, window=7, vector_size=200, sg=0, epochs=200)
        except Exception as e:
            print(f"Error training model on {name}: {e}")
            continue
        
        print(f"Similar words for '{test_names[i]}':")
        try:
            similar_words = model.wv.similar_by_word(test_names[i], topn=10)
            for result in similar_words:
                print(result[0], result[1])
        except KeyError:
            print(f"Word '{test_names[i]}' not in vocabulary.")
        
        print(f"Similar words for '{test_menpai[i]}':")
        try:
            similar_words = model.wv.similar_by_word(test_menpai[i], topn=10)
            for result in similar_words:
                print(result[0], result[1])
        except KeyError:
            print(f"Word '{test_menpai[i]}' not in vocabulary.")
