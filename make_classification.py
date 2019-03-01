from  learning_curve import plot_learning_cure
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.svm import LinearSVC

X, y = make_classification(1000, n_features=20, n_informative=2)
# print(X.shape) (1000, 20)
# print(y.shape)  (1000,)

# 在后面添加一个class的列，
df = DataFrame(np.hstack((X, y[:, None])), columns=list(range(20)) + ["class"])
#   DataFrame([data, index, columns, dtype, copy])
# print(df[:1])

# 看任意两点的相关度
_ = sb.pairplot(df[:50], vars=[8, 11, 12, 14, 19], hue="class", size=1.5)
# plt.show()

# 计算各特征之间的相关度
plt.figure(figsize=(12, 10))
_ = sb.heatmap(df, annot=False)
# plt.show()

plot_learning_cure(LinearSVC,'learning rate plot(C=10.0)',X,y,ylim=(0.8,1.01),
                   train_sizes=np.linspace(0.5,0.2,5))