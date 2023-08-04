from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as mp

iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris['data'].shape)
# print(iris['target'])
# print(iris['DESCR'])
# print(iris['frame'])
# print(iris['feature_names'])
x = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int_)

# print(x)
# print(y)

clf = LogisticRegression()
clf.fit(x, y)
qoi = clf.predict([[1.6]])
print(qoi)

x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = clf.predict_proba(x_new)
print(x_new)
mp.plot(x_new, y_prob[:, 1], "g-", label="virgin-ica")
mp.show()
