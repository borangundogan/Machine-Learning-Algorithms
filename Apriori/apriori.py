import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv("veriler/sepet.csv", header=None)

t = []

for i in range(0,7501):
    for j in range(0,20):
        t.append(str(veriler.values[i, j]))

print(t)
from apyori import apriori
rules = apriori(t,min_support=0.01, min_confidence=0.2, min_lift=3, min_length = 2)

print(list(rules))