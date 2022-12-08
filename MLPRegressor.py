import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('Iris.csv')

#q16
df['Species'] = df['Species'].replace(['Iris-setosa'], '0')
df['Species'] = df['Species'].replace(['Iris-versicolor'], '1')
df['Species'] = df['Species'].replace(['Iris-virginica'], '2')

columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = df.loc[:, columns]
y = df.loc[:, ['Species']]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=5)
model = MLPRegressor().fit(X_train, y_train.values.ravel())
print(model)
prediction= model.predict(X_test)
print( prediction)
print(metrics.r2_score(y_test, prediction))