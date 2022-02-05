import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

veriler = pd.read_csv("veriler/Ads_CTR_Optimisation.csv")

#print(veriler.describe())

# N = 10000
# d = 10
# sum = 0
# liste = [] 

# for n in range(0,N):
#     name = random.randrange(10)
#     liste.append(name)
#     price = veriler.values[n,name]
#     sum += price 

# plt.hist(liste)
# plt.show()

#Upper Bound

N = 10000 #10.000 tıklama
d = 10 #toplam 10 ilan
prices = [0] * d #başta bütün ilanların ödülü: 0 Ri[n]
sum_price = 0 # toplam ödül
clicks =  [0] * d #o ana kadarki tıklamalar: Ni[n]
liste = []

for n in range(0,N):
    name = 0 #seçilen ilan
    max_ucb = 0
    for i in range(0,d):
        if (clicks[i] > 0):
            ort = prices [i] / clicks [i]
            delta = math.sqrt((3/2) * math.log (n)/ clicks [i])
            ucb = ort + delta

        else:
            ucb =  N * 10

        if (max_ucb <ucb): #max'dan büyük bir ucb değeri çıktı
            max_ucb=ucb
            name = i
            
    liste.append(name)
    clicks[name] += 1
    price = veriler.values[n,name]
    prices[name] += price
    sum_price += price

print("Toplam Ödül: ")
print(sum_price)

plt.hist(liste)
plt.show()

