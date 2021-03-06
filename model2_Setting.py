import spacy
import torch


class Setting():
    def __init__(self):
        #数据准备部分：
        self.train_document_text = 'D:\\Research\\Data\\sumdata\\train\\valid.article.filter.txt'
        self.train_summary_text = 'D:\\Research\\Data\\sumdata\\train\\valid.title.filter.txt'

        self.train_cvs = 'D:\\Research\\Data\\dataFuntain_dataset\\train.csv'
        self.val_cvs = 'D:\\Research\\Data\\dataFuntain_dataset\\val.csv'
        self.test_cvs = 'D:\\Research\\Data\\dataFuntain_dataset\\test.csv'

        self.document_max = 250
        self.summary_min = 20
        self.summary_max = 75

        self.spacy_en = spacy.load('en_core_web_md')

        #训练的设置：
        self.BATCH_SIZE = 20
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ENC_EMB_DIM = 256//32
        self.DEC_EMB_DIM = 256//32
        self.ENC_HID_DIM = 512//32
        self.DEC_HID_DIM = 512//32
        self.ENC_DROPOUT = 0.5
        self.DEC_DROPOUT = 0.5

        self.N_EPOCHS = 300
        self.CLIP = 2
        self.lr = 0.01
        self.weight_decay = 0.005

        #结果
        self.result_path="D:\\Research\\My_model\\Model2\\result\\result_5.pt"

Setting=Setting()