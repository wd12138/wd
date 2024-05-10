import os
import jieba
import random
from gensim import corpora, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import numpy as np

def load_stopwords(path):
    with open(path, 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file.readlines()])
    return stopwords

def preprocess_text(text, stopwords):
    tokens = jieba.lcut(text)
    return [token for token in tokens if token not in stopwords and token.strip()]

def extract_paragraphs(tokens, num_paragraphs, k):
    selected_paragraphs = []
    for _ in range(num_paragraphs):
        if len(tokens) < k:
            continue  # 如果 tokens 的长度小于 k，则跳过当前段落的提取
        start_index = random.randint(0, len(tokens) - k)
        selected_paragraphs.append(tokens[start_index:start_index + k])
    return selected_paragraphs

def load_and_prepare_corpus(directory, stopwords, num_paragraphs, k):
    paragraphs = []
    labels = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8-sig', errors='ignore') as file:
            text = file.read()
            tokens = preprocess_text(text, stopwords)
            extracted_paragraphs = extract_paragraphs(tokens, num_paragraphs // len(os.listdir(directory)), k)
            paragraphs.extend(extracted_paragraphs)
            labels.extend([filename] * len(extracted_paragraphs))
    return paragraphs, labels

def preprocess_text_word(text, stopwords):
    tokens = jieba.lcut(text)
    return [token for token in tokens if token not in stopwords and token.strip()]

def preprocess_text_char(text, stopwords):
    # 将文本分割成字
    tokens = list(text)
    return [token for token in tokens if token not in stopwords and token.strip()]

def load_and_prepare_corpus_word(directory, stopwords, num_paragraphs, k):
    paragraphs = []
    labels = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8-sig', errors='ignore') as file:
            text = file.read()
            tokens = preprocess_text_word(text, stopwords)
            extracted_paragraphs = extract_paragraphs(tokens, num_paragraphs // len(os.listdir(directory)), k)
            paragraphs.extend(extracted_paragraphs)
            labels.extend([filename] * len(extracted_paragraphs))
    return paragraphs, labels

def load_and_prepare_corpus_char(directory, stopwords, num_paragraphs, k):
    paragraphs = []
    labels = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8-sig', errors='ignore') as file:
            text = file.read()
            tokens = preprocess_text_char(text, stopwords)
            extracted_paragraphs = extract_paragraphs(tokens, num_paragraphs // len(os.listdir(directory)), k)
            paragraphs.extend(extracted_paragraphs)
            labels.extend([filename] * len(extracted_paragraphs))
    return paragraphs, labels

def main(corpus_dir, stopwords_path, num_paragraphs, k, T):
    stopwords = load_stopwords(stopwords_path)
    selected_paragraphs_word, novel_labels_word = load_and_prepare_corpus_word(corpus_dir, stopwords, num_paragraphs, k)
    selected_paragraphs_char, novel_labels_char = load_and_prepare_corpus_char(corpus_dir, stopwords, num_paragraphs, k)

    if not selected_paragraphs_word or not selected_paragraphs_char:
        print("Error: Not enough paragraphs found.")
        return

    # 准备数据
    texts_word = [' '.join(paragraph) for paragraph in selected_paragraphs_word]
    texts_char = [''.join(paragraph) for paragraph in selected_paragraphs_char]

    # 创建词袋模型（以词为单位）
    vectorizer_word = CountVectorizer()
    X_word = vectorizer_word.fit_transform(texts_word)

    # 创建词袋模型（以字为单位）
    vectorizer_char = CountVectorizer(analyzer='char')
    X_char = vectorizer_char.fit_transform(texts_char)

    # 创建词典（以词为单位）
    dictionary_word = corpora.Dictionary([text.split() for text in texts_word])

    # 创建语料库（以词为单位）
    corpus_word = [dictionary_word.doc2bow(text.split()) for text in texts_word]

    # 创建词典（以字为单位）
    dictionary_char = corpora.Dictionary([list(text) for text in texts_char])

    # 创建语料库（以字为单位）
    corpus_char = [dictionary_char.doc2bow(list(text)) for text in texts_char]

    # LDA模型（以词为单位）
    lda_model_word = models.LdaModel(corpus_word, num_topics=T, id2word=dictionary_word, passes=10)

    # LDA模型（以字为单位）
    lda_model_char = models.LdaModel(corpus_char, num_topics=T, id2word=dictionary_char, passes=10)

    # 转换段落为 LDA 主题分布（以词为单位）
    X_lda_word = np.array([[prob for topic, prob in lda_model_word.get_document_topics(doc, minimum_probability=0)] for doc in corpus_word])

    # 转换段落为 LDA 主题分布（以字为单位）
    X_lda_char = np.array([[prob for topic, prob in lda_model_char.get_document_topics(doc, minimum_probability=0)] for doc in corpus_char])

    # 选择分类器（这里使用随机森林）
    classifier_word = RandomForestClassifier()
    classifier_char = RandomForestClassifier()

    # 交叉验证（以词为单位）
    scores_word = cross_val_score(classifier_word, X_lda_word, novel_labels_word, cv=10)
    print("以词为单位的交叉验证准确率：", np.mean(scores_word))

    # 交叉验证（以字为单位）
    scores_char = cross_val_score(classifier_char, X_lda_char, novel_labels_char, cv=10)
    print("以字为单位的交叉验证准确率：", np.mean(scores_char))

if __name__ == '__main__':
    main(r'C:\Users\qizh1\Desktop\jyxstxtqj_downcc.com', r'C:\Users\qizh1\Desktop\深度学习与nlp\nlp_homework2\stopwords.txt', 1000, 3000, 100)
