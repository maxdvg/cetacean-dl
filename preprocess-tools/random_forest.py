# Max Van Gelder
# 8/14/20

# Quick and somewhat dirty...
# Use random forest on preprocessed data from humpback and bowhead databases
# to train a random forest classifier to differentiate between the two based
# solely on feature extraction. Intended to proove that datasets are different
# enough that learning is possible.

# sys.argv[1] is the bowhead database
# sys.argv[2] is the humpback database

from sklearn.ensemble import RandomForestClassifier
import sqlite3
import sys

from database_tools import feature_list_to_string

# TODO: Automate this, save optimal cutoff threshold for positives in a seperate file
BH_CUTOFF = .65
HB_CUTOFF = .8


def chunk_data_getter(db_cursor, cutoff, feature_cols, bh=True):
    db_cursor.execute("SELECT {} FROM chunks WHERE RFPredForOne>{}".format(feature_list_to_string(feature_cols)
                                                                           , cutoff))
    data = db_cursor.fetchall()

    # Load the features into a numpy array that the random forest accepts
    features = np.empty([len(data), len(data[0])])
    for datum_idx, datum in enumerate(data):
        for feat_idx, feature in enumerate(datum):
            features[datum_idx][feat_idx] = feature

    if bh:
        y = np.zeros([len(features)])
    else:
        y = np.ones([len(features)])

    return features, y

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



if __name__ == "__main__":
    # connect to databases
    # Get handles for working with databases
    bh_conn = sqlite3.connect(sys.argv[1])
    bh_cursor = bh_conn.cursor()

    hb_conn = sqlite3.connect(sys.argv[2])
    hb_cursor = hb_conn.cursor()

    feature_columns = ["NumACP", "AvgACPSize", "AvgACPDensity"]

    bh_x, bh_y = chunk_data_getter(bh_cursor, BH_CUTOFF, feature_columns)
    hb_x, hb_y = chunk_data_getter(hb_cursor, HB_CUTOFF, feature_columns, bh=False)

    # Use 10000 total samples, with 8000 in training and 2000 in validation splits
    X = np.concatenate((bh_x[:8000], hb_x[:8000]))
    y = np.concatenate((bh_y[:8000], hb_y[:8000]))

    clf = RandomForestClassifier(max_depth=2, random_state=0)

    x_validation = np.concatenate((bh_x[8000:10000], hb_x[8000:10000]))
    ys = np.concatenate((bh_y[8000:10000], hb_y[8000:10000]))
    y_test = np.empty([len(ys), 2])
    for label_idx, label in enumerate(ys):
        if label == 0:
            y_test[label_idx][0] = 1
            y_test[label_idx][1] = 0
        else:
            y_test[label_idx][0] = 0
            y_test[label_idx][1] = 1

    clf.fit(X, y)
    y_score = clf.predict_proba(x_validation)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Bowhead Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print("I'm just here so I don't get fined")
