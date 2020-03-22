import pandas as pd
import numpy as np
import csv

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x = train.iloc[:, 2:].values
y = train.iloc[:, 1:2].values
x_test = test.iloc[:, 1:4].values

# for training set
from sklearn.impute import SimpleImputer
imputer_tr = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_tr = imputer_tr.fit(x[:, 2:])
x[:, 2:] = imputer_tr.transform(x[:, 2:])

# for test set
from sklearn.impute import SimpleImputer
imputer_te = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_te = imputer_te.fit(x_test[:, 2:])
x_test[:, 2:] = imputer_te.transform(x_test[:, 2:])


# encoding categorical data: train
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()
x[:, 1] = labelencoder_x.fit_transform(x[:, 1])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories="auto"), [1])],
    remainder="passthrough"
)
x = ct.fit_transform(x)

# encoding categorical data: test
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_xtest = LabelEncoder()
x_test[:, 1] = labelencoder_xtest.fit_transform(x_test[:, 1])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories="auto"), [1])],
    remainder="passthrough"
)
x_test = ct.fit_transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(x, y.ravel())

y_pred = classifier.predict(x_test)

print(type(y_pred))
