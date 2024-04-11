import os
import math
from collections import defaultdict

def calculate_char_frequencies(text):
    char_freqs = defaultdict(int)
    
    chinese_text = ''.join(filter(lambda char: '\u4e00' <= char <= '\u9fff', text)) #过滤非中文文字
    
    for char in chinese_text:
        char_freqs[char] += 1
    
    return char_freqs

def calculate_trigram_frequencies(text):
    trigram_freqs = defaultdict(int)
    
    chinese_text = ''.join(filter(lambda char: '\u4e00' <= char <= '\u9fff', text))#过滤非中文文字
    
    for i in range(len(chinese_text) - 2):
        trigram = chinese_text[i:i+3]
        trigram_freqs[trigram] += 1
    
    return trigram_freqs

def calculate_char_entropy(char_freqs):
    total_chars = sum(char_freqs.values())
    entropy = 0
    for freq in char_freqs.values():
        probability = freq / total_chars
        entropy -= probability * math.log2(probability)
    return entropy

def calculate_trigram_entropy(trigram_freqs):
    total_trigrams = sum(trigram_freqs.values())
    entropy = 0
    for freq in trigram_freqs.values():
        probability = freq / total_trigrams
        entropy -= probability * math.log2(probability)
    return entropy

def char_entropy_in_folder(folder_path):
    total_entropy = 0
    num_files = 0
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
        #if file_name=='越女剑.txt':
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
                char_freqs = calculate_char_frequencies(text)
                entropy = calculate_char_entropy(char_freqs)
                total_entropy += entropy
                num_files += 1
    
    average_entropy = total_entropy / num_files if num_files > 0 else 0
    return average_entropy

def trigram_entropy_in_folder(folder_path):
    total_entropy = 0
    num_files = 0
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
        #if file_name=='越女剑.txt':
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8',errors='ignore') as file:
                text = file.read()
                trigram_freqs = calculate_trigram_frequencies(text)
                entropy = calculate_trigram_entropy(trigram_freqs)
                total_entropy += entropy
                num_files += 1
    
    average_entropy = total_entropy / num_files if num_files > 0 else 0
    return average_entropy



folder_path = r'C:\Users\qizh1\Desktop\深度学习与nlp\中文语料库'
average_char_entropy = char_entropy_in_folder(folder_path)
print("Average character entropy in the folder:", average_char_entropy)
average_word_entropy = trigram_entropy_in_folder(folder_path)
print("Average trigram entropy in the folder:", average_word_entropy)
