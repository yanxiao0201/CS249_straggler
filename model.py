
import matplotlib.pyplot as plt
import numpy as np
import csv

#matplotlib inline
plt.rcParams['figure.figsize'] = (12, 10)


from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold as stk
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc,roc_curve, precision_recall_curve,confusion_matrix,classification_report

from scipy import interp

random_state = np.random.RandomState(42)

def plot_ROC_curve(classifier, X, y, pos_label=1, n_folds=5):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure("ROC")
    for i, (train, test) in enumerate(stk(y,shuffle=True,n_folds=n_folds)):
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
    for i, (train, test) in enumerate(stk(y,shuffle=True,n_folds=n_folds)):
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
        header = reader.next()[4:-1]

        for row in reader:
            feature = row[4:-1]
            feature = [float(i) for i in feature]
            label = int(row[-1])
            features.append(feature)
            labels.append(label)


#print header

total_features = np.asarray(features)
total_labels = np.asarray(labels)


#from sklearn.utils import shuffle
#df = shuffle(df)

#print col_name


print "Data visualization"

def draw_feature(x_fea, y_fea, total_features, total_labels, header):
    xdata = total_features[:,x_fea]
    ydata = total_features[:,y_fea]

    plt.figure("{} and {}".format(header[x_fea], header[y_fea]))

    xdata_0 = []
    xdata_1 = []
    ydata_0 = []
    ydata_1 = []

    for idx, i in enumerate(total_labels):
        if int(i) == 1:
            xdata_1.append(xdata[idx])
            ydata_1.append(ydata[idx])
        else:
            xdata_0.append(xdata[idx])
            ydata_0.append(ydata[idx])


    plt.scatter(xdata_0, ydata_0, label='Not straggler',color = 'b')
    plt.scatter(xdata_1, ydata_1, label='Straggler',color = 'r')
    plt.xlabel(header[x_fea])
    plt.ylabel(header[y_fea])
    plt.legend(loc="lower right")
    plt.savefig("{} and {}.png".format(header[x_fea], header[y_fea]))

'''
pairs = [0,1,2,-1,-2,-3,-4,-5,-7,-8]

for i in xrange(len(pairs)-1):
    for j in xrange(i+1,len(pairs)):
        draw_feature(pairs[i],pairs[j],total_features,total_labels,header)
'''



print "train blagging"
from blagging import BlaggingClassifier

rfc = RandomForestClassifier(class_weight = {1:50,0:1})
bbagging = BlaggingClassifier()

skf = StratifiedKFold(n_splits=4, shuffle = True)
for train_index, test_index in skf.split(total_features, total_labels):
        X_train, X_test = total_features[train_index],total_features[test_index]
        y_train, y_test = total_labels[train_index],total_labels[test_index]
        y_proba = bbagging.fit(X_train, y_train).predict_proba(X_test)

        for thres in np.arange(0.85,0.95,0.01):
            y_pred = []
            for entry in y_proba[:,1]:
                if entry > thres:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

            print "-----------------"
            print "data for thres = {}".format(thres)
            print confusion_matrix(y_test,y_pred)
            target_names = ['class 0', 'class 1']
            print classification_report(y_test,y_pred,target_names = target_names)

            print "-----------------"

        break


#plot_ROC_curve(bbagging, total_features, total_labels)
#plot_PR_curve(bbagging,total_features, total_labels)
