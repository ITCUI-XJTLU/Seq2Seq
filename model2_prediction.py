import torch
import spacy
import pandas as pd
import random
import os.path as op


from model2_function import generate_summary
from model2_getModel import DOCUMENT,SUMMARY
from model2_Setting import Setting


##加载模型
model=torch.load(Setting.result_path,map_location=torch.device('cpu'))
device=Setting.device

##准备测试数据
data_test = pd.read_csv(Setting.test_cvs,encoding='utf-8')
data_test = data_test[:10]
doc_sentence_list = data_test['document'].tolist()
sum_sentence_list = data_test['summary'].tolist()
sum1=['111','222']

##进行预测
generated_summary = []
for doc_sentence in doc_sentence_list:
    summary_words,attention = generate_summary(doc_sentence, DOCUMENT, SUMMARY, model, device)
    summary_sentence = (' ').join(summary_words)

    generated_summary.append(summary_sentence)
    print("document:"+doc_sentence)
    print("summary:"+summary_sentence)
    print()


