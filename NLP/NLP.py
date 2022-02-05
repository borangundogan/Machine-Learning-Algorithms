from nltk.util import pr
import pandas as pd
import numpy as np
import nltk
import re

path ="veriler/Restaurant_Reviews.csv"

comments = pd.read_csv(path, header=None,sep="delimiter")
print(comments) 

#stopwords temizleme
nltk.download("stopwords")
from nltk.corpus import stopwords as sw

from nltk.stem.porter import PorterStemmer
#ekleri silme
ps = PorterStemmer()

liste = []

for i in range (0,1000,1):
    comment = re.sub("[^a-zA-Z]","  ",comments["Review"][i])
    #harfleri küçüğe çevirme
    comment = comment.lower()
    #kelimeleri listeye haline çevirme
    comment = comment.split()
    comment = [ps.stem(word) 
        for word in comment 
            if not word in set(sw.words("english"))
    ]
    comment = " ".join(comment)
    liste.append(comment)
    
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2000)

#BAĞIMSIZ DEĞİŞKEN
X = cv.fit_transform(liste).toarray()

#BAĞIMLI DEĞİŞKEN
Y= comments.iloc[:,1].values()

from sklearn.model_selection  import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train,Y_train)

Y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test,Y_pred)
print(cm)