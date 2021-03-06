import os
from model2_function import make_cvs, clean_string

source_atitcle_folder='D:\\Research\\Data\\BBC News Summary\\BBC News Summary\\News Articles\\sport'
source_summary_folder='D:\\Research\\Data\\BBC News Summary\\BBC News Summary\\Summaries\\sport'
train_cvs = 'D:\\Research\\Data\\BBC News Summary\\BBC News Summary\\cvsFile\\sport\\train.csv'
val_cvs = 'D:\\Research\\Data\\BBC News Summary\\BBC News Summary\\cvsFile\\sport\\val.csv'
test_cvs = 'D:\\Research\\Data\\BBC News Summary\\BBC News Summary\\cvsFile\\sport\\test.csv'

list_artitle=os.listdir(source_atitcle_folder)
list_summary=os.listdir(source_summary_folder)

document1=[]
summary1=[]

for i in range(len(list_artitle)):
    aticle_path=os.path.join(source_atitcle_folder,list_artitle[i])
    summary_path=os.path.join(source_summary_folder,list_summary[i])

    artictle_content=""
    summary_content=""

    with open(aticle_path,'r') as file1 :
        for line in file1.readlines():
            artictle_content=artictle_content+line

    with open(summary_path,'r') as file2:
        for line in file2.readlines():
            summary_content=summary_content+line
    document1.append(summary_content)
    summary1.append(artictle_content)

print(len(document1))
print(len(summary1))
for i in range(len(document1)):
    print(i)

    document1[i]=clean_string(document1[i])
    summary1[i]=clean_string(summary1[i])



make_cvs(train_cvs,document1[0:round(len(document1)*0.8)],summary1[0:round(len(summary1)*0.8)])
#make_cvs(val_cvs,document1[round(len(document1)*0.8):round(len(document1)*0.995)],summary1[round(len(summary1)*0.8):round(len(document1)*0.995)])
#make_cvs(test_cvs,document1[round(len(document1)*0.995):len(document1)-1],summary1[round(len(summary1)*0.8):len(summary1)-1])
