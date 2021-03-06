import time
import torch.nn as nn
import torch.optim as optim
import torch
import math


from model2_getModel import model,SUMMARY,DOCUMENT,train_iter,val_iter
from model2_function import train as train0
from model2_function import evaluate,epoch_time
from model2_Setting import Setting

###训练部分
N_EPOCHS = Setting.N_EPOCHS
CLIP = Setting.CLIP
lr= Setting.lr
weight_decay= Setting.weight_decay
SUM_PAD_IDX = SUMMARY.vocab.stoi[SUMMARY.pad_token]

# 使用ignore_index参数，使得计算损失的时候不计算pad的损失
criterion = nn.CrossEntropyLoss(ignore_index = SUM_PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

# 训练
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train0(model, train_iter, optimizer, criterion,CLIP)
    valid_loss = evaluate(model, val_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)


    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')



#torch.save(model,"D:\\Research\\text-summariztaion\\text-summurization\\model2_result5.pt")
torch.save(model,Setting.result_path)
