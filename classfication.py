# -*- coding: utf-8 -*-



from sklearn.model_selection import train_test_split
import pandas as pd
#from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report



data_path = "metrics_1.csv"

print "load data
df = pd.read_csv(data_path, sep=',')
#from sklearn.utils import shuffle
#df = shuffle(df)
X = df.iloc[:, 2:-1]
y= df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#test_shape = y_test.shape[0]

#def compute_scores(y_test, predict):
#    print "accuracy_score ", accuracy_score(y_test, predict)
#    print "recall_score ", recall_score(y_test, predict)
#    print "precision_score ", precision_score(y_test, predict)



'''
from sklearn import svm
lin_svc = svm.LinearSVC()
lin_svc.fit(X_train, y_train)
svm_predicted = lin_svc.predict(X_test)
#compute_scores(y_test, svm_predicted)
target_names = ['0','1']
print(classification_report(y_test, svm_predicted, target_names=target_names))



from sklearn import linear_model
lr_norm2 = linear_model.LogisticRegression()
lr_norm2.fit(X_train, y_train)
lr_norm2_predicted = lr_norm2.predict(X_test)
#compute_scores(y_test, lr_norm2_predicted)
print(classification_report(y_test, lr_norm2_predicted, target_names=target_names))
'''

import bagging
bbagging = bagging.BlaggingClassifier()
bbagging.fit(X_train, y_train)
y_predict = bbagging.predict(X_test)
print(classification_report(y_test, y_predicted))
