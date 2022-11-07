# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def main():
    data = pd.read_table('D:\\Env\\vscode\\DeepLearning\\SVM\\watermelon.txt',delimiter=',')    
    x = pd.DataFrame({'密度':data['密度'],'含糖率':data['含糖率'],'触感':data['触感'],
                      '脐部':data['脐部'],
                      '纹理':data['纹理'],
                      '敲声':data['敲声'],
                      '根蒂':data['根蒂'],
                      '色泽':data['色泽']})
    x = x.values.tolist()    
    encoder = LabelEncoder()
    y = encoder.fit_transform(data['好瓜']).tolist()
    x,y = np.array(x),np.array(y)
    
    linear_svm = svm.SVC(C=0.5,kernel='linear')
    linear_svm.fit(x,y)
    
    
    gauss_svm = svm.SVC(C=0.5,kernel='rbf')
    gauss_svm.fit(x,y)
    
    svl = linear_svm.support_vectors_
    print(svl,end='\n');
    svg = gauss_svm.support_vectors_
    print(svg,end='\n')

if __name__=="__main__":
    main()
    
