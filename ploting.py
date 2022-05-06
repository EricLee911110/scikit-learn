import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification

plt.figure(figsize=(8,8))
plt.subplot(3,2,1)
plt.title("Mess up the color")

X1, Y1 = make_classification(
    n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1
)
arr=[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
plt.scatter(X1[:,0], X1[:,1], marker="o", c=arr, s=25, edgecolor="k")

plt.subplot(3,2,2)
plt.title("This is what the color should be")
plt.scatter(X1[:,0], X1[:,1], marker="o", c=Y1, s=25, edgecolor="k")

plt.show()

