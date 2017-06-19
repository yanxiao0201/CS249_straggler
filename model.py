import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

#matplotlib inline
plt.rcParams['figure.figsize'] = (12, 10)

from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn import cross_validation, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.svm import SVC
from sklearn.datasets.covtype import fetch_covtype

from sklearn.neural_network import MLPClassifier

from scipy import interp

random_state = np.random.RandomState(42)

def plot_ROC_curve(classifier, X, y, pos_label=1, n_folds=5):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    plt.figure("ROC")
    for i, (train, test) in enumerate(StratifiedKFold(y,shuffle=True,n_folds=n_folds)):
        #print (train,test)
        #print X[train]
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        #print classifier.feature_importances_
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    mean_tpr /= n_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_PR_curve(classifier, X, y, n_folds=5):
    """
    Plot a basic precision/recall curve.
    """
    plt.figure("PR")
    for i, (train, test) in enumerate(StratifiedKFold(y,shuffle=True,n_folds=n_folds)):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        precision, recall, thresholds = precision_recall_curve(y[test], probas_[:, 1],
                                                               pos_label=1)
        plt.plot(recall, precision, lw=1, label='PR fold %d' % (i,))
   #  clf_name = str(type(classifier))
   # clf_name = clf_name[clf_name.rindex('.')+1:]
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curve')
    plt.legend(loc="lower right")
    plt.show()




#data_paths = ["metrics_1.csv","metrics_2.csv","metrics_3.csv"]
data_paths = ["metrics_3.csv"]

labels = []

features = []
for data_path in data_paths:
    print "load data: {}".format(data_path)

    with open(data_path,'r') as csv_file:
        reader = csv.reader(csv_file)
        header = reader.next()

        for row in reader:
            feature = row[3:-2]
            feature = [float(i) for i in feature]
            label = int(row[0])
            features.append(feature)
            labels.append(label)


total_features = np.asarray(features)
total_labels = np.asarray(labels)

#from sklearn.utils import shuffle
#df = shuffle(df)

#print col_name


print "Data visualization"

def draw_feature(x_fea, y_fea, total_features, total_labels):
    xdata = total_features[:,x_fea]
    ydata = total_features[:,y_fea]

    plt.figure("Column #{} and #{}".format(x_fea+3, y_fea+3))

    xdata_0 =
    plt.plot(recall, precision, lw=1, label='PR fold %d' % (i,))





print "train blagging"
from blagging import BlaggingClassifier

bbagging = BlaggingClassifier()




plot_ROC_curve(bbagging, total_features, total_labels)
plot_PR_curve(bbagging,total_features, total_labels)
