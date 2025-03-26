import json
import pandas as pd

def outcsv(kokukoushi, fname, i, sh_name, fp):
    df = pd.read_excel(fname, sheet_name=i, usecols=[1, 3, 5, 6, 8], keep_default_na=False)
    line = 0
    for row in df.itertuples():
        if row[1]=='学部':
            line = row[0]
            break
    start = line + 3
    #cell_value = df.iloc[stop+1, 1]
    for row in df.itertuples():
        if start<row[0] and row[1]=='' and row[2]=='':
            break
        if start<=row[0]:
            str = json.dumps([kokukoushi, sh_name, row[1], row[2], row[3], row[4], row[5]], ensure_ascii=False)
            fp.write(str + '\n')
            #print(kokukoushi, sh_name, row[1], row[2], row[3], row[4], row[5])

if __name__ == '__main__':
    '''
    Excelファイルは次の文部科学省のサイトよからダウンロード
    https://www.mext.go.jp/a_menu/koutou/ichiran/mext_00026.html
    '''
    path = './'
    kokukoushi = ['国立', '公立', '私立']
    shiritsu = '20240607_mxt_daigakuc01_000036190_03-'    #+'1.xlsx'
    fname = ['20240607_mxt_daigakuc01_000036190_01.xlsx','20240607_mxt_daigakuc01_000036190_02.xlsx']
    for i in range(1, 9):
        x = shiritsu + str(i) + '.xlsx'
        fname.append(x)
    fp = open('daigaku.csv', 'wt')
    for n in range(len(fname)):
        j = n if n<2 else 2
        df = pd.read_excel(fname[n], sheet_name=None, usecols=[1, 3, 5, 6, 8])
        sh_name = list(df.keys())
        for i, name in enumerate(sh_name):
            outcsv(kokukoushi[j], fname[n], i, name, fp)
    fp.close()
