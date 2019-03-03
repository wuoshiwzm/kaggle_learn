from  learning_curve import plot_learning_curve
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.svm import LinearSVC
import time

X, y = make_classification(1000, n_features=20, n_informative=2)
# print(X.shape)
# print(y.shape)  (1000,)

# 在后面添加一个class的列，
df = DataFrame(np.hstack((X, y[:, None])), columns=list(range(20)) + ["class"])
# DataFrame([data, index, columns, dtype, copy])
# print(df[:1])

# 看任意两点的相关度
# _ = sb.pairplot(df[:50], vars=[8, 11, 12, 14, 19], hue="class", size=1.5)
# plt.show()

# 计算各特征之间的相关度
# plt.figure(figsize=(12, 10))
# _ = sb.heatmap(df, annot=False)
# plt.show()

# 减少特征
plot_learning_curve(LinearSVC(C=10.0), "LinearSVC(C=10.0) Features: 11&14", X[:, [11, 14]], y, ylim=(0.8, 1.0), train_sizes=np.linspace(.05, 0.1, 5))

# 增大训练集
# plot_learning_curve(LinearSVC(C=10),'learning rate plot(C=10.0)',X,y,ylim=(0.8,1.01),  train_sizes=np.linspace(.1,.992,5))
# plot_learning_curve(LinearSVC(C=1.0),'learning rate plot(C=10.0)',X,y,ylim=(0.8,1.01),  train_sizes=np.linspace(0.5,0.2,5))

# 遍历多种特征组合
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,f_classif
plot_learning_curve(Pipeline([("fs", SelectKBest(f_classif, k=2)), # select two features
                               ("svc", LinearSVC(C=10.0))]),
                    "SelectKBest(f_classif, k=2) + LinearSVC(C=10.0)", X, y,
                    ylim=(0.8, 1.0), train_sizes=np.linspace(.05, 0.2, 5))

#增强正则化作用(比如说这里是减小LinearSVC中的C参数)
plot_learning_curve(LinearSVC(C=0.1), "LinearSVC(C=0.1)", X, y, ylim=(0.8, 1.0), train_sizes=np.linspace(.05, 0.2, 5))

# 参数选择
from sklearn.grid_search import GridSearchCV
estm = GridSearchCV(LinearSVC(),
                   param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]})
plot_learning_curve(estm, "LinearSVC(C=AUTO)",
                    X, y, ylim=(0.8, 1.0),
                    train_sizes=np.linspace(.05, 0.2, 5))
print("Chosen parameter on 100 datapoints: %s" % estm.fit(X[:500], y[:500]).best_params_)


# 换成L1正则化
plot_learning_curve(LinearSVC(C=0.1, penalty='l1', dual=False), "LinearSVC(C=0.1, penalty='l1')", X, y, ylim=(0.8, 1.0), train_sizes=np.linspace(.05, 0.2, 5))
# 最后获得的权重：
estm = LinearSVC(C=0.1, penalty='l1', dual=False)
estm.fit(X[:450], y[:450])  # 用450个点来训练
print ("Coefficients learned: %s" % estm.coef_)
print ("Non-zero coefficients: %s" % np.nonzero(estm.coef_)[1])

# 结果
# Coefficients learned: [[ 0.          0.          0.          0.          0.          0.01857999
#    0.          0.          0.          0.004135    0.          1.05241369
#    0.01971419  0.          0.          0.          0.         -0.05665314
#    0.14106505  0.        ]]
# Non-zero coefficients: [5 9 11 12 17 18]

# 我们再随机生成一份数据[1000*20]的数据(但是分布和之前有变化)，重新使用LinearSVC来做分类。
#构造一份环形数据
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, random_state=2)
#绘出学习曲线
plot_learning_curve(LinearSVC(C=0.25),"LinearSVC(C=0.25)",X, y, ylim=(0.5, 1.0),train_sizes=np.linspace(.1, 1.0, 5))


# 看看数据的分布  数据没法线性分割
f = DataFrame(np.hstack((X, y[:, None])), columns = range(2) + ["class"])
_ = sb.pairplot(df, vars=[0, 1], hue="class", size=3.5)

# 解决方法 调整你的特征(找更有效的特征！！)
# 加入原始特征的平方项作为新特征
X_extra = np.hstack((X, X[:, [0]]**2 + X[:, [1]]**2))

plot_learning_curve(LinearSVC(C=0.25), "LinearSVC(C=0.25) + distance feature", X_extra, y, ylim=(0.5, 1.0), train_sizes=np.linspace(.1, 1.0, 5))

# 使用更复杂一点的模型(比如说用非线性的核函数)
from sklearn.svm import SVC
# note: we use the original X without the extra feature
plot_learning_curve(SVC(C=2.5, kernel="rbf", gamma=1.0), "SVC(C=2.5, kernel='rbf', gamma=1.0)",X, y, ylim=(0.5, 1.0), train_sizes=np.linspace(.1, 1.0, 5))




############################################################
#关于大数据样本集和高维特征空间
############################################################


# 生成大样本，高纬度特征数据
X, y = make_classification(200000, n_features=200, n_informative=25, n_redundant=0, n_classes=10, class_sep=2,
                           random_state=0)

# 用SGDClassifier做训练，并画出batch在训练前后的得分差
# 多层感知神经网络
from sklearn.linear_model import SGDClassifier

est = SGDClassifier(penalty="l2", alpha=0.001)
progressive_validation_score = []
train_score = []
for datapoint in range(0, 199000, 1000):
    X_batch = X[datapoint:datapoint + 1000]
    y_batch = y[datapoint:datapoint + 1000]
    if datapoint > 0:
        progressive_validation_score.append(est.score(X_batch, y_batch))
    est.partial_fit(X_batch, y_batch, classes=range(10))
    if datapoint > 0:
        train_score.append(est.score(X_batch, y_batch))

plt.plot(train_score, label="train score")
plt.plot(progressive_validation_score, label="progressive validation score")
plt.xlabel("Mini-batch")
plt.ylabel("Score")
plt.legend(loc='best')
plt.show()


############################################################
#降维
############################################################
# 大数据量下的可视化

#直接从sklearn中load数据集
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
print("Dataset consist of %d samples with %d features each" % (n_samples, n_features))

# 绘制数字示意图
n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
_ = plt.title('A selection from the 8*8=64-dimensional digits dataset')
plt.show()

# 随机投射
#import所需的package
from sklearn import (manifold, decomposition, random_projection)
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)

#定义绘图函数
from matplotlib import offsetbox
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    #设置图像范围
    plt.figure(figsize=(10, 10))
    # 等价于 plt.subplot(1,1,1)
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # 位置 X[i, 0], X[i, 1]
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 12})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

#记录开始时间
start_time = time.time()
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection of the digits (time: %.3fs)" % (time.time() - start_time))

# PCA降维
from sklearn import (manifold, decomposition, random_projection)
#TruncatedSVD 是 PCA的一种实现
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
#记录时间
start_time = time.time()
plot_embedding(X_pca,"Principal Components projection of the digits (time: %.3fs)" % (time.time() - start_time))


# 我们用到一个技术叫做t-SNE，sklearn的manifold对其进行了实现：
from sklearn import (manifold, decomposition, random_projection)
#降维
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
start_time = time.time()
X_tsne = tsne.fit_transform(X)
#绘图
plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time: %.3fs)" % (time.time() - start_time))



############################################################
#损失函数选择
############################################################

import numpy as np
import matplotlib.plot as plt
# 改自http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_loss_functions.html
xmin, xmax = -4, 4
xx = np.linspace(xmin, xmax, 100)
plt.plot([xmin, 0, 0, xmax], [1, 1, 0, 0], 'k-',
         label="Zero-one loss")
plt.plot(xx, np.where(xx < 1, 1 - xx, 0), 'g-',
         label="Hinge loss")
plt.plot(xx, np.log2(1 + np.exp(-xx)), 'r-',
         label="Log loss")
plt.plot(xx, np.exp(-xx), 'c-',
         label="Exponential loss")
plt.plot(xx, -np.minimum(xx, 0), 'm-',
         label="Perceptron loss")

plt.ylim((0, 8))
plt.legend(loc="upper right")
plt.xlabel(r"Decision function $f(x)$")
plt.ylabel("$L(y, f(x))$")
plt.show()






























































