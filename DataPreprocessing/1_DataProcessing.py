import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer

#download data
datas = pd.read_csv("C:/Users/red_h/OneDrive/Masaüstü/MachineLearning2/DataPreprocessing/eksikveriler.csv")
#print(datas)

#data preprocessing
boy = datas[["boy"]]

# missing values 
# sklearn (sci - kit learn) 

imputer  = SimpleImputer(missing_values=np.nan, strategy="mean")

age = datas.iloc[:,1:-1].values
#print(age)

#fit func use for learning
imputer = imputer.fit(age)

#transform func use for apply the learned informations.
age = imputer.transform(age)

print(age)

