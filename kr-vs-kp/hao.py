# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

import warnings
import datetime
import os

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from imblearn.over_sampling import SMOTE
import itertools
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
import random

# with open("agaricus-lepiota.df") as f:
# headname = pd.read_table("agaricus-lepiota.df", sep=',')
df = pd.read_table("kr-vs-kp.data", sep=',', header=None, names=
['bkblk','bknwy','bkon8','bkona','bkspr','bkxbq','bkxcr','bkxwp','blxwp','bxqsq','cntxt','dsopp','dwipd',
 'hdchk','katri','mulch','qxmsq','r2ar8','reskd','reskr','rimmx','rkxwp','rxmsq','simpl','skach','skewr',
 'skrxp','spcop','stlmt','thrsk','wkcti','wkna8','wknck','wkovl','wkpos','wtoeg','label'])

print(df)
df  = pd.get_dummies(df)
print(df.columns)

X = df[['bkblk_f', 'bkblk_t', 'bknwy_f', 'bknwy_t', 'bkon8_f', 'bkon8_t',
       'bkona_f', 'bkona_t', 'bkspr_f', 'bkspr_t', 'bkxbq_f', 'bkxbq_t',
       'bkxcr_f', 'bkxcr_t', 'bkxwp_f', 'bkxwp_t', 'blxwp_f', 'blxwp_t',
       'bxqsq_f', 'bxqsq_t', 'cntxt_f', 'cntxt_t', 'dsopp_f', 'dsopp_t',
       'dwipd_g', 'dwipd_l', 'hdchk_f', 'hdchk_t', 'katri_b', 'katri_n',
       'katri_w', 'mulch_f', 'mulch_t', 'qxmsq_f', 'qxmsq_t', 'r2ar8_f',
       'r2ar8_t', 'reskd_f', 'reskd_t', 'reskr_f', 'reskr_t', 'rimmx_f',
       'rimmx_t', 'rkxwp_f', 'rkxwp_t', 'rxmsq_f', 'rxmsq_t', 'simpl_f',
       'simpl_t', 'skach_f', 'skach_t', 'skewr_f', 'skewr_t', 'skrxp_f',
       'skrxp_t', 'spcop_f', 'spcop_t', 'stlmt_f', 'stlmt_t', 'thrsk_f',
       'thrsk_t', 'wkcti_f', 'wkcti_t', 'wkna8_f', 'wkna8_t', 'wknck_f',
       'wknck_t', 'wkovl_f', 'wkovl_t', 'wkpos_f', 'wkpos_t', 'wtoeg_n',
       'wtoeg_t']]
# record = pd.read_csv("./input/credit_record.csv", encoding='utf-8')
Y = df['label_won']

print(X)
print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=10086)
print("shape")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# s = random.sample(range(2000),300)
# X_test = X_test[s]
# y_test = y_test[s]

#KNN
model = KNeighborsClassifier(weights='distance')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("\nKNeighbors")
print(y_predict)
KNeighborsPre = y_predict
print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test, y_predict)))

#GBDT
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("\nGBDT")
print(y_predict)
GBDTPre = y_predict
print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test, y_predict)))

#Bagging
model = BaggingClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("\nBagging")
print(y_predict)
BaggingPre = y_predict
print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test, y_predict)))

# LogisticRegression
model = LogisticRegression(C=0.8,
                           random_state=0,
                           solver='lbfgs')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("\nLogisticRegression")
print(y_predict)
LogisticRegressionPre = y_predict
print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test, y_predict)))

# sns.set_style('white')
class_names = ['0', '1']

# DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=12,
                               min_samples_split=8,
                               random_state=1024)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("\nDecisionTree")
print(y_predict)
DecisionTreePre = y_predict
print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test, y_predict)))

model = RandomForestClassifier(n_estimators=250,
                               max_depth=12,
                               min_samples_leaf=16
                               )
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("\nRandomForest")
print(y_predict)
RandomForestPre = y_predict
print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test, y_predict)))

model = svm.SVC(C=0.8,
                kernel='linear')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("\nSVM")
print(y_predict)
SVMPre = y_predict
print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test, y_predict)))

model = LGBMClassifier(num_leaves=31,
                       max_depth=8,
                       learning_rate=0.2,
                       n_estimators=250,
                       subsample=0.8,
                       colsample_bytree=0.8
                       )

lbl = preprocessing.LabelEncoder()
# X_train['wkphone'] = lbl.fit_transform(X_train['wkphone'].astype(str))
# X_test['wkphone'] = lbl.fit_transform(X_test['wkphone'].astype(str))
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("\nLGBM")
print(y_predict)
LGBMPre = y_predict
print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test, y_predict)))

# def plot_importance(classifer, x_train, point_size=25):
#     '''plot feature importance'''
#     values = sorted(zip(x_train.columns, classifer.feature_importances_), key=lambda x: x[1] * -1)
#     imp = pd.DataFrame(values, columns=["Name", "Score"])
#     imp.sort_values(by='Score', inplace=True)
#     sns.scatterplot(x='Score', y='Name', linewidth=0,
#                     df=imp, s=point_size, color='red').set(
#         xlabel='importance',
#         ylabel='features')


# plot_importance(model, X_train, 20)

# model.booster_.feature_importance(importance_type='gain')

model = XGBClassifier(max_depth=12,
                      n_estimators=250,
                      min_child_weight=8,
                      subsample=0.8,
                      learning_rate=0.2,
                      seed=42)

model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("\nXGB")
print(y_predict)
XGBPre = y_predict
print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test, y_predict)))

# plot_importance(model, X_train, 20)

model = CatBoostClassifier(iterations=50,
                           learning_rate=1,
                           od_type='Iter',
                           verbose=25,
                           depth=16,
                           random_seed=42)

model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("\nCatBoost")
print(y_predict)
CatBoostPre = y_predict
print('CatBoost Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test, y_predict)))

# (34596, 29)
# (14828, 29)
# (34596,)
# (14828,)
LogisticRegressionPre
DecisionTreePre
RandomForestPre
SVMPre
LGBMPre
XGBPre
CatBoostPre
preAll = np.zeros((X_test.shape[0], 10))

for i in range((X_test.shape[0])):
    preAll[i, 0] = LogisticRegressionPre[i]
    preAll[i, 1] = DecisionTreePre[i]
    preAll[i, 2] = RandomForestPre[i]
    preAll[i, 3] = SVMPre[i]
    preAll[i, 4] = LGBMPre[i]
    preAll[i, 5] = XGBPre[i]
    preAll[i, 6] = CatBoostPre[i]
    preAll[i, 7] = KNeighborsPre[i]
    preAll[i, 8] = GBDTPre[i]
    preAll[i, 9] = BaggingPre[i]
print(preAll)
#
starttime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
path = "./" + str(starttime)
os.mkdir(path)
np.savetxt(path + '/preAll.txt', preAll)
np.savetxt(path + '/X_train.txt', X_train)
np.savetxt(path + '/X_test.txt', X_test)
np.savetxt(path + '/y_train.txt', y_train)
np.savetxt(path + '/y_test.txt', y_test)
