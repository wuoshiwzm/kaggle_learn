import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
from sklearn.svm import LinearSVC
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5 , train_sizes=np.linspace(.1, 1.0, 5)):
    """
        画出data在某模型上的learning curve.
        参数解释
        ----------
        estimator : 你用的分类器。
        title : 表格的标题。
        X : 输入的feature，numpy类型
        y : 输入的target vector
        ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
        cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
        """
    plt.figure()
    #
    # train_sizes_abs : array, shape (n_unique_ticks,), dtype int Numbers of training examples that has been used to generate the learning curve.
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes)
    print(train_sizes, train_scores, test_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # plt.fill_between(x, y1, y2, facecolor="lightgray")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')

    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std,alpha=0.1,color="r")

    plt.plot(train_sizes,train_scores_mean,'o-',color="r",label="Training score")
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label="Cross-vali score")

    plt.xlabel('Training examples')
    plt.ylabel('score')
    plt.legend(loc='best')
    # 网络
    plt.grid('on')
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()































































































