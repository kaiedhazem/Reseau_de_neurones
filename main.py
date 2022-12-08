# importing 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as s
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.model_selection import train_test_split
# q1
df = pd.read_csv('Iris.csv')
#q2
print(df.head(10))
#q3
print(df.shape)
#q4
g = s.pairplot(df,hue="Species", vars=["PetalLengthCm","PetalWidthCm"])
plt.show()
#q5
df['Species'] = df['Species'].replace(['Iris-setosa'], '0')
df['Species'] = df['Species'].replace(['Iris-versicolor'], '1')
df['Species'] = df['Species'].replace(['Iris-virginica'], '2')
#q6
print(df.head(10))
#q7
columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = df.loc[:, columns]
y = df.loc[:, ['Species']]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=5)
#q8
print(X_train.head(10))
print(X_test.head(10))

#q9
mlp =MLPClassifier(solver='lbfgs', alpha =1e-05, hidden_layer_sizes=(3,3), epsilon=0.07, max_iter=150)
mlp.fit(X_train, y_train)
prediction= mlp.predict(X_test)
print(prediction)
# q10
print(metrics.accuracy_score(prediction, y_test))
# q11
cmap='PuRd'
pp_matrix_from_data(y_test.values, prediction)
print('Error rate of the Multi Layer Perceptron MLP :' + str(1-metrics.accuracy_score(y_test,prediction)))
#q14
params= [ 
    {
        "solver":"sgd",
        "learning_rate":"constant",
        "learning_rate_init":0.2,
        "max_iter": 150,  
    },

    {
        "solver":"sgd",
        "learning_rate":"constant",
        "learning_rate_init":0.7,
        "max_iter": 300, 
    },
     {
        "solver":"sgd",
        "learning_rate":"invscaling",
        "learning_rate_init": 0.2,
        "max_iter": 300,
    },

     {
        "solver":"sgd",
        "learning_rate":"invscaling",
        "learning_rate_init": 0.7,
        "max_iter": 150,  
    },
    {
        "solver":"adam",
        "learning_rate_init": 0.01,
        "max_iter": 300,    
    },
]

labels=[
    "constant learning-rate_0.2",
    "constant learning-rate_0.7",
    "invscaling learning-rate_0.2",
    "invscaling learning-rate_0.7",
    "adam",
]

plot_args=[
    {"c": "red", "linestyle": "-"},
    {"c": "green", "linestyle": "-"},
    {"c": "blue", "linestyle": "-"},
    {"c": "red", "linestyle": "--"},
    {"c": "green", "linestyle": "--"},
]
mlps = []
for label, param in zip(labels, params):
    print('training : %s' % label)
    mlp=MLPClassifier(random_state=0, **param)
    mlp.fit(X_train, y_train)
    mlps.append(mlp)
    print("training set score : %f" % mlp.score(X_train, y_train))

for mlp,label,args in zip(mlps, labels, plot_args):
    plt.plot(mlp.loss_curve_)
    plt.title(" %s " %label, fontsize=12)
    plt.show()