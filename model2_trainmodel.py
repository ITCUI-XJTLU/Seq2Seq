import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# encoder的输入为原文，输出为hidden_state，size需要设置
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        # 定义embedding层，直接使用torch.nn.Embedding函数
        #input_dim代表词表的大小，emb_dim定义每个词的维度
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # 定义rnn层，这里使用torch.nn.GRU
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        # 定义一个全连接层，用来将encoder的输出转换成decoder输入的大小
        #Since bidirection=true from above, the dimision of output will double
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        # 定义dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, doc, doc_len):
        # doc = [doc len, batch size]
        # doc_len = [batch size]

        embedded = self.dropout(self.embedding(doc))

        # embedded = [doc len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, doc_len)

        packed_outputs, hidden = self.rnn(packed_embedded)

        # packed_outputs 包含了每个RNN中每个状态的输出，如图中的h1,h2,h3...hn
        # hidden只有最后的输出hn

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs已经做了填充，但是后续计算attention会规避掉填充的信息

        '''n layers代表gru的层数，这里只使用的单层，因此n layer为1'''
        # outputs = [doc len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden 包含了每一层的最后一个状态的输出，是前后向交替排列的 [第一层前向RNN最后一个状态输出, 第一层后向RNN最后一个状态输出
        #  第二层前向RNN最后一个状态输出, 第一层后向RNN最后一个状态输出, ...]
        # outputs 仅包含了每个状态最后一层的的输出，且是前后向拼接的

        # hidden [-2, :, : ] 前向RNN的最后一个状态输出
        # hidden [-1, :, : ] 后向RNN的最后一个状态输出

        # 用一个全连接层将encoder的输出表示转换成decoder输入的大小
        # tanh为非线性激活函数
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [doc len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [doc len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        doc_len = encoder_outputs.shape[0]

        # 对decoder的状态重复doc_len次，用来计算和每个encoder状态的相似度
        hidden = hidden.unsqueeze(1).repeat(1, doc_len, 1)

        #转置
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, doc len, dec hid dim]
        # encoder_outputs = [batch size, doc len, enc hid dim * 2]

        # 使用全连接层计算相似度
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, doc len, dec hid dim]

        # 转换尺寸为[batch, doc len]的形式作为和每个encoder状态的相似度
        attention = self.v(energy).squeeze(2)

        # attention = [batch size, doc len]

        # 规避encoder里pad符号，将这些位置的权重值降到很低
        attention = attention.masked_fill(mask == 0, -1e10)

        # 返回权重
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [doc len, batch size, enc hid dim * 2]
        # mask = [batch size, doc len]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask)

        # a = [batch size, doc len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, doc len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, doc len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions 在decoder为1的情况比较多, 所以:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # output和hidden应该是相等的，output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), a.squeeze(1)




class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, doc_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.doc_pad_idx = doc_pad_idx
        self.device = device

    def create_mask(self, doc):
        mask = (doc != self.doc_pad_idx).permute(1, 0)
        return mask

    def forward(self, doc, doc_len, sum, teacher_forcing_ratio=0.5):
        # doc = [doc len,batch size]
        # doc_len = [doc len,batch size]
        # sum = [batch size, sum len]
        # teacher_forcing_ratio 是使用teacher forcing的概率

        batch_size = doc.shape[1]
        sum_len = sum.shape[0]
        sum_vocab_size = self.decoder.output_dim

        # 定义一个tensor来储存每一个生成的单词序号
        outputs = torch.zeros(sum_len, batch_size, sum_vocab_size).to(self.device)

        # encoder_outputs是encoder所有的输出状态
        # hidden这是encoder整体的输出
        encoder_outputs, hidden = self.encoder(doc, doc_len)

        # 输入的第一个字符为<sos>
        input = sum[0, :]

        # 构建一个mask矩阵，包含训练数据原文中pad符号的位置
        mask = self.create_mask(doc)

        # mask = [batch size, doc len]

        for t in range(1, sum_len):
            # decoder 输入 前一步生成的单词embedding, 前一步状态hidden, encoder所有状态以及mask矩阵
            # 返回预测全连接层的输出和这一步的状态
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

            # 把output的信息存储在之前定义的outputs里
            outputs[t] = output

            # 生成一个随机数，来决定是否使用teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # 获得可能性最高的单词序号作为生成的单词
            top1 = output.argmax(1)

            # 如果使用teacher forcing则用训练数据相应位置的单词
            # 否则使用生成的单词 作为下一步的输入单词
            input = sum[t] if teacher_force else top1

        return outputs



