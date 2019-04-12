

```python
# kaggle的titanic入门比赛，训练集和测试集都下载在本文件的同一目录
```


```python
# 数据总体分析
import pandas as pd
train = pd.read_csv('train.csv')
train.head()
# Survived => 1生存 0遇难
# PassengerId => 乘客ID
# Survived => 获救情况（1为获救，0为未获救）
# Pclass => 乘客等级(1/2/3等舱位)
# Name => 乘客姓名
# Sex => 性别
# Age => 年龄
# SibSp => 堂兄弟/妹个数
# Parch => 父母与小孩个数
# Ticket => 船票信息
# Fare => 票价
# Cabin => 客舱
# Embarked => 登船港口
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
# Age, Cabin缺失比较严重， Embarked缺失两个值，特征处理时要进行填充
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    


```python
train.describe()
# Survived的平均值为0.383，我们的预测准确率至少达到
# 1 - 0.383 = 0.617 才能接受，否则为什么不全部预测为 0 呢
# 一半乘客都是三等舱的 20到40岁的人占了一半 最大年龄为80
# SibSp和Parch两极分化严重，两个变量相似，考虑合并为一个
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
train[['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']].describe()
# name是唯一的，应该跟生存关系不大 男性占了577/891=65% 
# 像Cabin，Ticket等文本类，我们要做的是提取信息，简化信息
# Ticket没有缺失但只有681个值，说明有人共用一张船票，可以把共用船票的人数构造为一个特征值
# Cabin的204个值中大多数不一样，类太多考虑合并
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Sex</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>891</td>
      <td>2</td>
      <td>681</td>
      <td>147</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Levy, Mr. Rene Jacques</td>
      <td>male</td>
      <td>1601</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>577</td>
      <td>7</td>
      <td>4</td>
      <td>644</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 基于前面分析进行特征处理
# preprocess函数是通用的处理，之后还要针对模型进行特定处理
class PreProcessor:
    def fit(self, data):
        # 复用训练集的age_mean，fare_mean
        self.age_mean = data['Age'].mean()
        self.fare_mean = data['Fare'].mean()
        return self
        
    def tranform(self, data):
        # 均值填充Age
        data['Age'].fillna(self.age_mean, inplace=True)
        # 用‘S’填充
        data['Embarked'].fillna('S', inplace=True)
        # 测试数据的Fare有一个空值
        data['Fare'].fillna(self.fare_mean, inplace=True)
        # 合并SibSp和Patch列
        data['FamilyNum'] = data.SibSp + data.Parch
        # Cabin有值的为一类，无值的为一类
        data['CabinClass'] = data.Cabin.map(lambda x: 0 if pd.isna(x) else 1)
        # 多少人共用一张船票做为一个数值特征
        data['TicketNum'] = data.Ticket.map(dict(data.groupby('Ticket').PassengerId.count()))
        # 我们舍弃PassengerId,Name，其他的列已经做了处理
        data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], inplace=True)
        # 对分类特征one-hot,pandas虽然把Pclass当成数值特征，
        # 但我们应该清楚它属于分类特征
        return pd.get_dummies(
                    data,
                    columns=['Pclass', 'Sex', 'CabinClass', 'Embarked']
                    )

# 开始建立模型
# 再import一次pandas，是为了该代码块能作为一个py文件
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# 关掉一些版本变动提示
import warnings
warnings.filterwarnings('ignore')

# 读取数据集
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
result = {'PassengerId': test['PassengerId']}

# 处理特征
prepro = PreProcessor().fit(train)

train = prepro.tranform(train)
train_x = train.drop(columns=['Survived'])
train_x_std = normalize(train_x) #逻辑回归还需要标准化
train_y = train['Survived']

test_x = prepro.tranform(test)

# 初始化模型
lr = LogisticRegression()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()

# 输出10折交叉验证平均得分
print ('LinearRegression ', np.mean(cross_val_score(lr, train_x_std, train_y, cv=10)))
print ('DecisionTree ', np.mean(cross_val_score(dtc, train_x, train_y, cv=10)))
print ('RandomForest ', np.mean(cross_val_score(rfc, train_x, train_y, cv=10)))
```

    LinearRegression  0.6915903416184316
    DecisionTree  0.7846402224492113
    RandomForest  0.8014561343774826
    


```python
# lr得分惨不忍睹，我们调一下参数
for c in [0.01, 0.1, 1, 10, 100, 200, 300, 400]:
    lr = LogisticRegression(penalty='l1', C=c)
    print ('LinearRegression ', np.mean(cross_val_score(lr, train_x_std, train_y, cv=10)))
```

    LinearRegression  0.6161701282487799
    LinearRegression  0.6735115196912951
    LinearRegression  0.7274716263761207
    LinearRegression  0.7846396549767336
    LinearRegression  0.7925053909885371
    LinearRegression  0.793628986494155
    LinearRegression  0.793628986494155
    LinearRegression  0.793628986494155
    


```python
# 调整决策树深度
for d in [1, 2, 3, 6, 7, 8, 9]:
    dtc = DecisionTreeClassifier()
    print ('DecisionTree ', np.mean(cross_val_score(dtc, train_x, train_y, cv=10)))
```

    DecisionTree  0.7846654749744637
    DecisionTree  0.7914076154806492
    DecisionTree  0.7812824877993416
    DecisionTree  0.7801208716377256
    DecisionTree  0.7869254341164453
    DecisionTree  0.7869004653274316
    DecisionTree  0.7914073317444104
    


```python
# 调整随机森林大小
for n in [120, 300, 500, 800]:
    rfc = RandomForestClassifier(n_estimators=n, max_depth=5)
    print ('RandomForest ', np.mean(cross_val_score(rfc, train_x, train_y, cv=10)))
```

    RandomForest  0.8092089433662467
    RandomForest  0.8148269208943366
    RandomForest  0.8114686187719895
    RandomForest  0.8103450232663716
    


```python
# 根据交叉验证平均得分以及调参结果，
# 最终选择训练n_etimators参数为300的rfc模型
rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(train_x, train_y)
# 预测并保存结果
test_y = rfc.predict(test_x)
result['Survived'] = test_y
result = pd.DataFrame(result)
result.to_csv('submission.csv',index=False)
```


```python
# 总结
# kaggle好像在中国没有服务器，总是出问题，submission.csv上传不了
# 不过没关系，过程比结果重要（这句话在数据挖掘方面怎么有点难说出口(￣▽￣)"）
# 数据分析没有标准答案，重要的是要有自己的一套方法，然后不断在实践中改进
# 最后，在目录中还提供了一份完整的py文件
```
