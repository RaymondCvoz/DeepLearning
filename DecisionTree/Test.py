import pandas as pd

df = pd.read_csv('H:\RC\DeepLearning\DecisionTree\data\car_copy.data')
dataset = df.values.tolist()

fp = open('H:\\RC\\DeepLearning\\DecisionTree\\tmp.out','w')

print(dataset,file=fp)