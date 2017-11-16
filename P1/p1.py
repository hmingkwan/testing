import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, \
     GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from mlxtend.classifier import StackingClassifier
from boruta import BorutaPy
from xgboost import XGBClassifier

# Read traindata, trainlabel and testdata
train = pd.read_csv('/Users/hokming/Desktop/MSBD6000B/Project1/traindata.csv', header=None)
label = pd.read_csv('/Users/hokming/Desktop/MSBD6000B/Project1/trainlabel.csv', header=None)
test = pd.read_csv('/Users/hokming/Desktop/MSBD6000B/Project1/testdata.csv', header=None)

# name the column
train.columns = ['feature' + str(col) for col in train.columns]
label.columns = ['label']
test.columns = ['feature' + str(col) for col in test.columns]

# split the train data for validation
X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.3, random_state=0)

# example for getting the feature importances
etc = ExtraTreesClassifier(random_state=0)
etc.fit(X_train, y_train.ix[:, -1])

importances = etc.feature_importances_

std = np.std([etc.feature_importances_ for etc in etc.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# KNN
clf1 = KNeighborsClassifier(n_neighbors=5)

# Random Forest
rf = RandomForestClassifier(random_state=0)
X_train_2 = X_train[["feature54", "feature51", "feature23", "feature6", "feature15", "feature20",
                     "feature52", "feature22", "feature4", "feature56", "feature55",
                     "feature26", "feature25", "feature16", "feature18", "feature24"
                    ]]
X_train_2 = preprocessing.normalize(X_train_2)
rf = rf.fit(X_train_2, y_train.ix[:, -1])

# Gaussian Naive Bayes
clf3 = GaussianNB()

# Ada Boost Classifier
abc = AdaBoostClassifier(random_state=0)

X_train_3 = X_train[["feature16", "feature52", "feature51", "feature15", "feature56", "feature55",
                     "feature24", "feature20", "feature17", "feature45", "feature7",
                     "feature4", "feature54", "feature26"
                    ]]
X_train_3 = preprocessing.normalize(X_train_3)
abc = abc.fit(X_train_3, y_train.ix[:, -1])

# Extra Trees Classifier
etc = ExtraTreesClassifier(random_state=0)
X_train_1 = X_train[["feature51", "feature20", "feature55", "feature15", "feature6", "feature22",
                     "feature52", "feature54", "feature26", "feature23", "feature24",
                     "feature18", "feature2", "feature4", "feature56", "feature25", "feature5", "feature36",
                    ]]
X_train_1 = preprocessing.normalize(X_train_1)
etc = etc.fit(X_train_1, y_train.ix[:, -1])

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=0)

X_train_4 = X_train[["feature51", "feature24", "feature52", "feature45", "feature55", "feature54",
                     "feature15", "feature56", "feature18", "feature26", "feature6",
                     "feature4", "feature20", "feature7", "feature23", "feature41",
                     "feature19", "feature36", "feature16"
                    ]]
X_train_4 = preprocessing.normalize(X_train_4)
gbc = gbc.fit(X_train_4, y_train.ix[:, -1])

#  Extreme Gradient Boosting Classifier
xgb = XGBClassifier(random_state=0)

X_train_5 = X_train[["feature24", "feature51", "feature54", "feature26", "feature45", "feature6",
                     "feature15", "feature52", "feature55", "feature49", "feature18",
                     "feature4", "feature41", "feature5", "feature16", "feature11",
                    "feature22", "feature56", "feature44", "feature36", "feature2"
                    ]]

X_train_5 = preprocessing.normalize(X_train_5)
xgb = xgb.fit(X_train_5, y_train.ix[:, -1])


# Support Vector Classifier
svc = SVC()

# LDA
ldq = LinearDiscriminantAnalysis()

# QDA
qda = QuadraticDiscriminantAnalysis()

# Logistic Regression Classifier
lr = LogisticRegression()

# group all the classifiers with meta classifier - Logistic Regression Classifier
sclf = StackingClassifier(classifiers=[ clf1, rf, clf3, abc, etc, gbc, xgb, svc, lda, qda],
                          meta_classifier=lr)

# grid search parameters
params1 = {
    'kneighborsclassifier__n_neighbors': [1, 5],
    'randomforestclassifier': [
        {'n_estimators': np.arange(50, 500, 50)},
        {'max_depth': np.arange(1, 10)},
        {'criterion': ['gini', 'entropy']}],
    'adaboostclassifier': [
       {'n_estimators': np.arange(50, 500, 50)}],
    'extratreesclassifier': [
        {'n_estimators': np.arange(50, 500, 50)},
        {'max_depth': np.arange(1, 10)},
        {'criterion': ['gini', 'entropy']}],
    'gradientboostingclassifier': [
        {'n_estimators': np.arange(50, 500, 50)},
        {'learning_rate': np.arange(0.01, 0.1, 0.01)},
        {'max_depth': np.arange(1, 10)}],
     'svc': [
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.0001]}],
    'xgbclassifier': [
        {'learning_rate': np.arange(0.01, 0.1, 0.01)},
        {'n_estimators': np.arange(50, 500, 50)},
        {'max_depth': np.arange(1, 10)}],
    'meta-logisticregression__C': [0.1, 1.0, 10.0]
}

# grid search with 10 times stratified fold
grid = GridSearchCV(estimator=sclf,
                    param_grid=params1,
                    cv=StratifiedKFold(n_splits=10),
                    refit=True)
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

# fit the stacking model
grid.fit(X_train, y_train.ix[:, -1])

# validation
stack_preds = grid.predict_proba(X_test)[:,1]

# performance of validation
stack_performance = roc_auc_score(y_test, stack_preds)
print 'Stacking: Area under the ROC curve = {}'.format(stack_performance) #0.978624208304

# predict test data
result = grid.predict(test)

# save the predicted test result to csv
np.savetxt("/Users/hokming/Desktop/MSBD6000B/Project1/project1_20386486.csv", result, delimiter=",")