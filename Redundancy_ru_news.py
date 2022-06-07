from __future__ import annotations
import os
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pymorphy2 import MorphAnalyzer
parser = MorphAnalyzer()
import math
import nltk
nltk.download('punkt')
from nltk.probability import FreqDist
import re
import numpy as np
import time
from tqdm import tqdm
import json
from scipy import special
from typing import Optional, Union

#!/usr/bin/env python
#-*- encoding: utf-8 -*-

def get_text_not_lemm(data):
    text_ru = ['<s>'] 
    text_zh = ['<s>']
    body = data.body
    ses_ru  = body.find_all('se', {'lang':'ru'})
    for se in ses_ru:
        string = se.text
        string = nltk.word_tokenize(string)
        for word in string:
            if word != '\n' and word != '':
                #word = parser.parse(word)[0].normal_form
                #word = word.lower()
                text_ru.append(word)
    text_ru.append('</s>')

                
    ses_zh  = body.find_all('se', {'lang':'zh'})
    for se in ses_zh:
        se= str(se)
        ses = re.findall(r'(<\/ana>)(.*?)(<\/w>)(.*?)(<w>)?(.?)', se)
        for se in ses:
            text_zh.append(se[1])
            if se[-1] != '<':
                text_zh.append(se[-1])
    text_zh.append('</s>')
            
    return text_ru, text_zh


# Русский язык. Токены и их частотность

corpus_ru = open('all_texts_ru.txt', 'r', encoding = 'utf-8').read().split()

corpus_ru_token = []
for token in corpus_ru:
    if token:
        corpus_ru_token.append(token)

corpus_ru_token_freq = dict(FreqDist(corpus_ru_token))

with open('corpus_ru_token_freq.json', 'w', encoding='utf-8') as file:
    file.write(json.dumps(corpus_ru_token_freq, ensure_ascii=False))


# Русский язык. Биграммы, их частотность и предсказуемость

corpus_ru_bigram = list(nltk.bigrams(corpus_ru))

corpus_ru_bigram_freq = dict(FreqDist(corpus_ru_bigram))

with open('corpus_ru_bigram_freq.json', 'w', encoding='utf-8') as file:
    file.write(json.dumps(str(corpus_ru_bigram_freq), ensure_ascii=False))
    

# Предсказуемость

corpus_ru_bigram_predict = {}

for bigram_item, bigram_value in corpus_ru_bigram_freq.items():
    for token_item, token_value in corpus_ru_token_freq.items():
        if token_item == bigram_item[0]:
            corpus_ru_bigram_predict[bigram_item] = bigram_value / token_value

with open('corpus_ru_bigram_predict.json', 'w', encoding='utf-8') as file:
    file.write(json.dumps(str(corpus_ru_bigram_predict), ensure_ascii=False))


# Равнозначная предсказуемость

#corpus_ru_bigram_max_predict = {}

#for bigram_item, bigram_value in corpus_ru_bigram_freq.items():
    #max_predictability = 1 / len(corpus_ru_bigram_freq)
    #corpus_ru_bigram_max_predict[bigram_item] = max_predictability


# Русский язык. Новостные тексты

PATH = './texts/rus-zho/news_texts'
news_xml_ru = []
for root, dirs, files in os.walk(PATH):
    for file in files:
        with open (os.path.join(root, file),'r', encoding='utf-8') as open_file:
            news_xml_ru.append(open_file.read())

news_texts_ru = {}
for xml in news_xml_ru:
    data = BeautifulSoup(xml)
    head = data.head
    title = head.find('title').text
    text_ru, text_ru = get_text_not_lemm(data)
    news_texts_ru[title] = text_ru

# Предсказуемость и равнозначная предсказуемость

news_texts_ru_bigram_predict = {}
for label, news_text_ru in tqdm(news_texts_ru.items()):
    news_text_ru_bigram = list(nltk.bigrams(news_text_ru))
    news_text_ru_bigram_predict = {}
    news_text_ru_bigram_max_predict = {}
    for bigram in news_text_ru_bigram:
        for bigram_predict_item, bigram_predict_value in corpus_ru_bigram_predict.items():
            if bigram == bigram_predict_item:
                news_text_ru_bigram_predict[bigram] = bigram_predict_value
    news_texts_ru_bigram_predict[label] = news_text_ru_bigram_predict

with open('news_texts_ru_bigram_predict.txt', 'w', encoding='utf-8') as file:
    file.write(str(news_texts_ru_bigram_predict))


# Энтропия

news_texts_ru_particular_entropy = {}
for label, news_text_ru_bigram_predict in news_texts_ru_bigram_predict.items():
    news_text_ru_particular_entropy = []
    pk = np.asarray(list(news_text_ru_bigram_predict.values()))
    pk = 1.0 * pk / np.sum(pk, axis=int(0), keepdims=True)
    vec = special.entr(pk)
    S = np.sum(vec, axis=int(0))
    S /= np.log(2)
    news_text_ru_particular_entropy.append(S)
    news_texts_ru_particular_entropy[label] = news_text_ru_particular_entropy

with open('news_texts_ru_particular_entropy.txt', 'w', encoding='utf-8') as file:
    file.write(str(news_texts_ru_particular_entropy))

# Максимальная энтропия

news_texts_ru_max_entropy = {}
for label, news_text_ru_bigram_max_predict in news_texts_ru_bigram_predict.items():
  news_text_ru_max_entropy = []
  max_S = np.log2(len(corpus_ru_token_freq))
  news_text_ru_max_entropy.append(max_S)
  news_texts_ru_max_entropy[label] = news_text_ru_max_entropy

with open('news_texts_ru_max_entropy.txt', 'w', encoding='utf-8') as file:
    file.write(str(news_texts_ru_max_entropy))

# Энтропия, которая нужна для избыточности

news_texts_ru_entropy = {}
for label, particular_entropy in news_texts_ru_particular_entropy.items():
    news_text_ru_entropy = []
    for label_max, max_entropy in news_texts_ru_max_entropy.items():
      if label == label_max:
        for p in particular_entropy:
            for m in max_entropy:
                entropy = float(p) / float(m)
    news_text_ru_entropy.append(entropy)
    news_texts_ru_entropy[label] = news_text_ru_entropy

with open('news_texts_ru_entropy.txt', 'w', encoding='utf-8') as file:
    file.write(str(news_texts_ru_entropy))

# Избыточность

news_texts_ru_redundancy = {}
for label, entropy in news_texts_ru_entropy.items():
    news_text_ru_redundancy = []
    for e in entropy:
        redundancy = 1 - e
        news_text_ru_redundancy.append(redundancy)
        news_texts_ru_redundancy[label] = news_text_ru_redundancy

with open('news_texts_ru_redundancy.txt', 'w', encoding='utf-8') as file:
    file.write(str(news_texts_ru_redundancy))

with open('news_texts_ru_redundancy.json', 'w', encoding='utf-8') as file:
    file.write(json.dumps(news_texts_ru_redundancy, ensure_ascii=False))

