from re import X
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


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

#numpy series convert to Dataframe

result = pd.DataFrame(data=country, index= range(22), columns= ["fr", "tr", "us"])
#print(result)

result2 = pd.DataFrame(data=age, index= range(22), columns=["boy", "kilo", "yas"])
#print(result2)

result3 = pd.DataFrame(data=gender, index= range(22), columns=["cinsiyet"])
#print(result3)

#data concat
concat1 = pd.concat([result, result2], axis=1)
#print(concat1)

concat2 = pd.concat([concat1, result3], axis=1)
#print(concat2)

#TEST-TRAIN
#Dependency variables -> y (gender prediction)
#UnDependency variables -> x (country, age, size, weight)

x_train, x_test, y_test, y_train = train_test_split(concat1, result3, test_size=0.33,random_state=0)
#print(x_train)

#Standart Scaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

print(X_train)
print(X_test)
print(Y_train)
print(Y_test)