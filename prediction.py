#! flask/bin/python

from pprint import pprint
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from time import time
from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import scale
from IPython.display import display

data = pd.read_csv('/home/nazaif/py_server/datasets/final_dataset.csv')

data = data[data.MW > 3]

data.drop(['Unnamed: 0', 'HomeTeam', 'AwayTeam', 'Date', 'MW', 'HTFormPtsStr', 'ATFormPtsStr', 'FTHG', 'FTAG',
           'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HomeTeamLP', 'AwayTeamLP', 'DiffPts', 'HTFormPts', 'ATFormPts',
           'HM4', 'HM5', 'AM4', 'AM5', 'HTLossStreak5', 'ATLossStreak5', 'HTWinStreak5', 'ATWinStreak5',
           'HTWinStreak3', 'HTLossStreak3', 'ATWinStreak3', 'ATLossStreak3'], 1, inplace=True)

n_matches = data.shape[0]
n_features = data.shape[1] - 1
n_homewins = len(data[data.FTR == 'H'])

display(data.head())

win_rate = (float(n_homewins) / (n_matches)) * 100

print "Total number of matches: {}".format(n_matches)
print "Number of features: {}".format(n_features)
print "Number of matches won by home team: {}".format(n_homewins)
print "Win rate of home team: {:.2f}%".format(win_rate)

# scatter_matrix(data[['HTGD','ATGD','HTP','ATP','DiffFormPts','DiffLP']], figsize=(10,10))

X_all = data.drop(['FTR'], 1)
y_all = data['FTR']

cols = [['HTGD', 'ATGD', 'HTP', 'ATP', 'DiffLP']]
for col in cols:
    X_all[col] = scale(X_all[col])

X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')


def preprocess_features(X):
    output = pd.DataFrame(index=X.index)
    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)
        output = output.join(col_data)
    return output


X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                    test_size=50,
                                                    random_state=2,
                                                    stratify=y_all)


def train_classifier(clf, X_train, y_train):
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    print "Trained model in {:.4f} seconds".format(end - start)


def predict_labels(clf, features, target):
    start = time()
    y_pred = clf.predict(features)

    end = time()
    print "Made predictions in {:.4f} seconds.".format(end - start)

    return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    train_classifier(clf, X_train, y_train)

    f1, acc = predict_labels(clf, X_train, y_train)
    print f1, acc
    print "F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc)

    f1, acc = predict_labels(clf, X_test, y_test)
    print "F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc)


clf_C = xgb.XGBClassifier(seed=82)

train_predict(clf_C, X_train, y_train, X_test, y_test)
print ''

parameters = {'learning_rate': [0.1],
              'n_estimators': [40],
              'max_depth': [3],
              'min_child_weight': [3],
              'gamma': [0.4],
              'subsample': [0.8],
              'colsample_bytree': [0.8],
              'scale_pos_weight': [1],
              'reg_alpha': [1e-5]
              }

clf = xgb.XGBClassifier(seed=2)

f1_scorer = make_scorer(f1_score, pos_label='H')

grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)

grid_obj = grid_obj.fit(X_train, y_train)

clf = grid_obj.best_estimator_
print clf

f1, acc = predict_labels(clf, X_train, y_train)
print "F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc)

parameters = {'learning_rate': [0.03],
              'n_estimators': [20],
              'max_depth': [5],
              'min_child_weight': [5],
              'gamma': [0.2],
              'subsample': [0.8],
              'colsample_bytree': [0.8],
              'scale_pos_weight': [1],
              'reg_alpha': [1e-2]
              }

clf = xgb.XGBClassifier(seed=2)

f1_scorer = make_scorer(f1_score, pos_label='H')

grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)

grid_obj = grid_obj.fit(X_all, y_all)

clf = grid_obj.best_estimator_
print clf

# clf.predict(x_test[1])
f1, acc = predict_labels(clf, X_train, y_train)
# print "F1 score and accuracy score for training 2nd set: {:.4f} , {:.4f}.".format(f1 , acc)

f1, acc = predict_labels(clf, X_test, y_test)
print "F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc)
