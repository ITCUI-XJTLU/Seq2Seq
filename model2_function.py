import torch
import spacy
from model2_Setting import Setting
import csv
import re


spacy_en=Setting.spacy_en
#构建分词函数
def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text) ]


def make_cvs(cvs_path,document,summary):
    with open(cvs_path, 'w', newline="") as c:
        filenames0 = ['document', 'summary']
        writer = csv.DictWriter(c, filenames0)
        writer.writeheader()
        for index in range(len(document)):
            writer.writerow({"document": document[index], "summary": summary[index]})


def clean_string(text):
    text = re.sub("\.", '\b', str(text)).lower()
    text=re.sub("(\\t)",'',str(text)).lower()
    text=re.sub("(\\r)",'',str(text)).lower()
    text=re.sub("(\\n)",'',str(text)).lower()
    text=re.sub("-",'',str(text)).lower()
    text=re.sub("#",'',str(text)).lower()
    text=re.sub("<unk>",'',str(text)).lower()
    text=re.sub("(\.\.+)",'\.',str(text)).lower()
    text=re.sub("(\b\b+)","\b",str(text)).lower()

    return text


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        doc, doc_len = batch.document
        sum = batch.summary

        optimizer.zero_grad()

        output = model.forward(doc, doc_len, sum)

        # sum = [sum len, batch size]
        # output = [sum len, batch size, output dim]
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        sum = sum[1:].view(-1)

        # sum = [(sum len - 1) * batch size]
        # output = [(sum len - 1) * batch size, output dim]

        loss = criterion(output, sum)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        if i>20:break

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            doc, doc_len = batch.document
            sum = batch.summary

            output = model(doc, doc_len, sum, 0)  # 验证时不使用teacher forcing

            # sum = [sum len, batch size]
            # output = [sum len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            sum = sum[1:].view(-1)

            # sum = [(sum len - 1) * batch size]
            # output = [(sum len - 1) * batch size, output dim]

            loss = criterion(output, sum)

            epoch_loss += loss.item()
            if i>20:break

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def generate_summary(doc_sentence, doc_field, sum_field, model, device, max_len=50):
    # 将模型置为验证模式
    model.eval()

    # 对原文分词
    nlp = spacy.load('en_core_web_md')
    #     nlp = spacy.load('en_core_web_md')
    tokens = [token.text.lower() for token in nlp(doc_sentence)]

    # 为原文加上起始符号<sos>和结束符号<eos>
    tokens = [doc_field.init_token] + tokens + [doc_field.eos_token]

    # 将字符转换成序号
    doc_indexes = [doc_field.vocab.stoi[token] for token in tokens]

    # 转换成可以gpu计算的tensor
    doc_tensor = torch.LongTensor(doc_indexes).unsqueeze(1).to(device)

    doc_len = torch.LongTensor([len(doc_indexes)]).to(device)

    # 计算encoder
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(doc_tensor, doc_len)

    mask = model.create_mask(doc_tensor)

    # 生成摘要的一个单词 <sos>

    sum_indexes = [sum_field.vocab.stoi[sum_field.init_token]]

    # 构建一个attention tensor，存储每一步的attention
    attentions = torch.zeros(max_len, 1, len(doc_indexes)).to(device)

    for i in range(max_len):

        sum_tensor = torch.LongTensor([sum_indexes[-1]]).to(device)

        # 计算每一步的decoder
        with torch.no_grad():
            output, hidden, attention = model.decoder(sum_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention

        pred_token = output.argmax(1).item()

        # 如果出现了 <eos> 则直接结束计算
        if pred_token == sum_field.vocab.stoi[sum_field.eos_token]:
            break

        sum_indexes.append(pred_token)

    # 把序号转换成单词
    sum_tokens = [sum_field.vocab.itos[i] for i in sum_indexes]

    return sum_tokens[1:], attentions[:len(sum_tokens) - 1]

