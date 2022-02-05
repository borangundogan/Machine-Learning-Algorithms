import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random


veriler = pd.read_csv("veriler/musteriler.csv")

#print(veriler[:10])

X = veriler.iloc[:,2:4].values
#print(X)

from sklearn.cluster import KMeans

results = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i , init="k-means++", random_state=123).fit(X)    
    results.append(kmeans.inertia_)
    
print(results)


kmeans = KMeans(n_clusters=4 , init="k-means++" ).fit(X)
print(kmeans.cluster_centers_)

y_pred_kmeas = kmeans.fit_predict(X)

print(y_pred_kmeas)

plt.scatter(X[y_pred_kmeas==0,0], X[y_pred_kmeas==0,1], s=100,c="red")
plt.scatter(X[y_pred_kmeas==1,0], X[y_pred_kmeas==1,1], s=100,c="blue")
plt.scatter(X[y_pred_kmeas==2,0], X[y_pred_kmeas==2,1], s=100,c="green")
plt.scatter(X[y_pred_kmeas==3,0], X[y_pred_kmeas==3,1], s=100,c="yellow")

plt.title("K-means")
#plt.plot(range(1,10) ,results)
plt.show()

#Hiyerarşik Bölütleme

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3 ,affinity="euclidean", linkage="ward")

y_pred = ac.fit_predict(X)

print(y_pred)

plt.scatter(X[y_pred==0,0], X[y_pred==0,1], s=100,c="red")
plt.scatter(X[y_pred==1,0], X[y_pred==1,1], s=100,c="blue")
plt.scatter(X[y_pred==2,0], X[y_pred==2,1], s=100,c="green")
plt.scatter(X[y_pred==3,0], X[y_pred==3,1], s=100,c="yellow")

plt.title("HC")
plt.show()

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X,method="median"))
plt.show()