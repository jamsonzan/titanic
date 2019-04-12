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

# 根据交叉验证平均得分以及调参结果，最终选择训练n_etimators参数为300的rfc模型
rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(train_x, train_y)
# 预测并保存结果
test_y = rfc.predict(test_x)
result['Survived'] = test_y
result = pd.DataFrame(result)
result.to_csv('submission.csv',index=False)

#特征处理类
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
        # 对分类特征one-hot,pandas虽然把Pclass当成数值特征，但我们应该清楚它属于分类特征
        return pd.get_dummies(
                    data,
                    columns=['Pclass', 'Sex', 'CabinClass', 'Embarked']
                    )
