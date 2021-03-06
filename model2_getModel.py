import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, BucketIterator, TabularDataset, Iterator
import spacy

import math
import time

from model2_trainmodel import Encoder, Decoder,Attention,Seq2Seq
from model2_function import epoch_time,evaluate, tokenize
from model2_Setting import Setting

'''
# 全局初始化配置参数。固定随机种子，使得每次运行的结果相同
SEED = 22

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
'''


train_cvs=Setting.train_cvs
val_cvs=Setting.val_cvs
test_cvs=Setting.test_cvs

BATCH_SIZE=Setting.BATCH_SIZE
###数据预处理
#加载英语库
spacy_en=Setting.spacy_en
#设置预处理函数格式
DOCUMENT = Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True,init_token='<sos>', eos_token='<eos>')
SUMMARY = Field(sequential=True, tokenize=tokenize, lower=True,init_token='<sos>', eos_token='<eos>')
fields=[("document",DOCUMENT),("summary",SUMMARY)]

train=TabularDataset(path=train_cvs,format="CSV",fields=fields,skip_header=True)
val=TabularDataset(path=val_cvs,format="CSV",fields=fields,skip_header=True)
test=TabularDataset(path=test_cvs,format="CSV",fields=fields,skip_header=True)


#构建数据迭代器
device=Setting.device
train_iter = BucketIterator(train, batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.document), sort_within_batch=True)

val_iter =  BucketIterator(val, batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.document), sort_within_batch=True)

test_iter =  BucketIterator(test, batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.document), sort_within_batch=True)


#构建词组列表
DOCUMENT.build_vocab(train, min_freq = 2)
SUMMARY.build_vocab(train, min_freq = 2)

print("前一百词：")
print(DOCUMENT.vocab.itos[:100])


##实例化模型：
INPUT_DIM = len(DOCUMENT.vocab)
OUTPUT_DIM = len(SUMMARY.vocab)
ENC_EMB_DIM = Setting.ENC_EMB_DIM
DEC_EMB_DIM = Setting.DEC_EMB_DIM
ENC_HID_DIM = Setting.ENC_HID_DIM
DEC_HID_DIM = Setting.DEC_HID_DIM
ENC_DROPOUT = Setting.ENC_DROPOUT
DEC_DROPOUT = Setting.DEC_DROPOUT
DOC_PAD_IDX = DOCUMENT.vocab.stoi[DOCUMENT.pad_token]

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model=Seq2Seq(enc, dec, DOC_PAD_IDX, device).to(device)

print(model)
print()
print()











