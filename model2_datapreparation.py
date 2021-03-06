from model2_Setting import Setting
from model2_function import clean_string,make_cvs


#数据集位置
train_document_text=Setting.train_document_text
train_summary_text=Setting.train_summary_text

#目标的CVS文件
train_cvs=Setting.train_cvs
val_cvs=Setting.val_cvs
test_cvs=Setting.test_cvs



document0=[]
summary0=[]
document_max=Setting.document_max
summary_min=Setting.summary_min
summary_max=Setting.summary_max
with open(train_document_text,'r') as doc:
    for line in doc.readlines():
        document0.append(clean_string(line))
with open(train_summary_text,'r') as sum:
    for line in sum.readlines():
        summary0.append(clean_string(line))

document1=[]
summary1=[]

for i in range(len(document0)):
    #print("document:"+str(len(document0[i]))+"\t\t\t"+"summury:"+str(len(summary0[i])))
    if len(document0[i])<=document_max and len(summary0[i])>=summary_min and len(summary0[i])<=summary_max:
    #if len(summary0[i]) >= summary_min:
        document1.append(document0[i])
        summary1.append(summary0[i])




print("document1 length:"+str(len(document1)))
print("summury1 length:"+str(len(summary1)))

'''
for i in range(100):
    print(document1[i])
    print(summary1[i])
    print()
'''

make_cvs(train_cvs,document1[0:round(len(document1)*0.8)],summary1[0:round(len(summary1)*0.8)])
make_cvs(val_cvs,document1[round(len(document1)*0.8):round(len(document1)*0.995)],summary1[round(len(summary1)*0.8):round(len(document1)*0.995)])
make_cvs(test_cvs,document1[round(len(document1)*0.995):len(document1)-1],summary1[round(len(summary1)*0.8):len(summary1)-1])







