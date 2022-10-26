import pandas as pd

df = pd.read_csv('./data/car.data')

i = 0
for index,series in df.iterrows():
    i += 1
    print(index,end='\n')
    print(series)
    if i == 1:
        break
    

dflist = df.values.tolist()

print(dflist)