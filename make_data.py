import pandas as pd

df = pd.read_csv('train_hate_speech.tsv', sep='\t')
df_2000 = df.iloc[2001:4000, :]

df_2000 = df_2000[['comments', 'hate']]
df_2000.reset_index(inplace=True, drop=True)
df_2000 = df_2000[(df_2000['hate'] == 'hate') | (df_2000['hate'] == 'offensive')]
len(df_2000)

Q_list = []
for i in range(len(df)):
    print(df_2000['comments'][i])
    Q = str(input('대답에 해당하는 질문 입력:'))
    Q_list.append(Q)

# B에 대한 대답
# 400~800
df = pd.read_csv('AIhub.csv', index_col=0, names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
df = df.iloc[400:800, 0:2]
df

df.reset_index(inplace=True, drop=True)
df.columns
# B에 대한 대답
# 이자식아
Q_list = []
for i in range(len(df)):
    print(df['B'][i])
    Q = str(input('해당하는 대답 입력:'))
    Q_list.append(Q)

len(Q_list)
df['C'] = Q_list

del df['A']

df.rename(columns={'B':'Q', 'C':'A'}, inplace=True)
df.to_csv('AIhub_y.csv')


ni_list = []
ni_index = []
for i in range(len(df)):
    if df['A'][i].find('니.') > 0:
        ni_index.append(i)
        ni_list.append(df['A'][i])
len(ni_list)

len(set(ni_list))
ni_index

def get_rough_dic_2():
    myword = {
        '탑니다.' : '타',
        '싶습니다.' : '싶어',
        '있습니다.' : '있어',
        '높아집니다.' : '높아져',
        '알겠습니다.' : '알겠어',
        '듯합니다.' : '듯 해',
        '좋습니다.' : '좋아',
        '같습니다.' : '같아',
        '괜찮습니다.' : '괜찮아',
        '기도합니다.' : '기도해',
        '자신이랍니다.' : '너야',
        '됩니다.' : '돼',
        '바랍니다.' : '바래',
        '한답니다.' : '해',
        '합니다.' : '해',
        '취직이나라니다.' : '취직이야',
        '믿습니다.' : '믿어',
        '하겠습니다.' : '할게',
        '씁니다.' : '써',
        '기원합니다.' : '바래',
        '얻습니다.' : '얻어',
        '끝났습니다.' : '끝났어',
        '부족합니다.' : '부족해',
        '아니랍니다.' : '아니야',
        '안됩니다.' : '안돼',
        '발바니다.' : '바래',
        '기대합니다.' : '기대해',
        '지냈답니다.' : '지냈어',
        '없답니다.' : '없어',
        '추천합니다.' : '추천해', 
        '축하합니다.' : '축하해',
        '감사합니다.' : '고마워',
        '노력하겠습니다.' : '노력할게',
        '사랑합니다.' : '사랑해',
        '존중합니다.' : '존중할게',
        '덜해집니다.' : '덜해져',
        '시작됩니다.' : '시작돼',
        '흘러갑니다.' : '흘러가',
        '좋아합니다.' : '좋아해',
        '넓답니다.' : '넓어'
    }
    return myword
banmal_dic = get_rough_dic_2()

df['A'][ni_index]
nida = []
for i in ni_list:
    i = i.replace('니.', '니다.')
    nida.append(i)

for i in ni_index:
    if df['A'][i].find('니.') > 0:
        df['A'][i] = df['A'][i].replace('니.', '니다.')

set(df['A'][ni_index])

eos = []
for i in nida:
    eos.append(i.split(' ')[-1])

set(eos)


for i, j in zip(banmal_dic.keys(), banmal_dic.values()):
    for k in ni_index:
        if df['A'][k].find(i) > 0:
            df['A'][k] = df['A'][k].replace(i, j)
            print(k)
        else:
            pass
df['A'][k]
df['A'][ni_index]

df.to_csv('df_banmal_f.csv', index=False)

# 바퀴 나쁜말하기
import pandas as pd
df = pd.read_csv('df_baqui.csv', index_col=0)
df = df[6000:12000][::-1]
df.reset_index(inplace=True, drop=True)
df

A_list = []
for i in range(len(df)):
    print(df['Q'][i])
    A = str(input('%d번째 질문에 해당하는 대답 입력:'%(i+1)))
    A_list.append(A)
    if A == 'quit':
        break

    del A_list[-1]
    len(A_list)

    df_100 = df.iloc[0:104, :]
    df_100['A'] = A_list
    df_100.to_csv('baqui_y_1.csv', index=False)


from tkinter import Tk, font
root = Tk()
font.families()
