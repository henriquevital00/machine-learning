import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('./50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],
                       remainder='passthrough')

X = np.array(ct.fit_transform(X))

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(
    np.concatenate(
        (Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

# Making a single prediction (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

print(regressor.coef_)
print(regressor.intercept_)

# Profit=86.6×Dummy State 1−873×Dummy State 2+786×Dummy State 3−0.773×R&D Spend+0.0329×Administration+0.0366×Marketing Spend+42467.53
