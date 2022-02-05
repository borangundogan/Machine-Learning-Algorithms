import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn import preprocessing

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


country = datas.iloc[:,0: 1].values
#print(country)

#Label Encoding

label_encoding = preprocessing.LabelEncoder()

country[:,0] = label_encoding.fit_transform(country)

#print(country)

#One Hot Encoding

Ohe = preprocessing.OneHotEncoder()

country = Ohe.fit_transform(country).toarray()
#print(country)

# Label Binarizer
label_binarizer = preprocessing.LabelBinarizer()

gender = datas.iloc[:,-1:].values
#print(gender)

gender[:,-1:] = label_binarizer.fit_transform(gender)

#print(gender)

#Concat to data each other

result = pd.DataFrame(data=country, index= range(22), columns= ["fr", "tr", "us"])
#print(result)

result2 = pd.DataFrame(data=age, index= range(22), columns=["boy", "kilo", "yas"])
#print(result2)

result3 = pd.DataFrame(data=gender, index= range(22), columns=["cinsiyet"])
#print(result3)

concat1 = pd.concat([result, result2], axis=1)
#print(concat1)

concat2 = pd.concat([concat1, result3], axis=1)
print(concat2)

