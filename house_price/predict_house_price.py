# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_dt = pd.read_csv('./input/train.csv',index_col=0)
test_dt = pd.read_csv('./input/test.csv',index_col=0)
# print(type(test_dt))

# 在数据预处理时首先可以对偏度比较大的数据用log1p函数
# 进行转化，使其更加服从高斯分布
price = pd.DataFrame({
    "price":train_dt['SalePrice'],'log(pric+1)':np.log1p(train_dt['SalePrice'])
})

plt.imshow(price, cmap='gray')
plt.colorbar()
plt.show()
# price.hist()
