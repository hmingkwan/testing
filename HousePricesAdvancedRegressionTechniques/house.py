import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer, RobustScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
import keras
from keras.models import Sequential
from keras.layers import Dense
from scipy.stats import norm

class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, mod, meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def fit(self, X, y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))

        for i, model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X, y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index, i] = renew_model.predict(X[val_index])

        self.meta_model.fit(oof_train, y)
        return self

    def predict(self, X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1)
                                      for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)


def display_distribution(pd, feature, name):
    plt.figure()
    sns.distplot(pd[feature].dropna(), fit=norm);
    (mu, sigma) = norm.fit(pd[feature].dropna())

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    plt.savefig(name)

def process_data():

    train=pd.read_csv('/Users/hokming/Downloads/all/train.csv')
    test=pd.read_csv('/Users/hokming/Downloads/all/test.csv')

    # plot the data for investigation: histogram
    train.hist(bins=50, figsize=(25, 18))
    plt.savefig('house_plot/hist')

    # YearBuilt vs SalePrice
    plt.figure(figsize=(15,8))
    sns.boxplot(train.YearBuilt, train.SalePrice)
    plt.savefig('house_plot/SalePrice_YearBuilt')

    # plot outliers
    plt.figure(figsize=(12, 6))
    plt.scatter(x=train.GrLivArea, y=train.SalePrice)
    plt.xlabel("GrLivArea", fontsize=13)
    plt.ylabel("SalePrice", fontsize=13)
    plt.ylim(0, 800000)
    plt.savefig('house_plot/Outliers')

    # remove outliters
    train.drop(train[(train["GrLivArea"]>4000)&(train["SalePrice"]<300000)].index,inplace=True)

    full=pd.concat([train,test], ignore_index=True)

    full.drop(['Id'],axis=1, inplace=True)

    # distribution of SalePrice before and after normalization
    display_distribution(train, 'SalePrice', 'house_plot/skewed_saleprice')
    train["SalePrice"] = np.log1p(train["SalePrice"])
    display_distribution(train, 'SalePrice', 'house_plot/normal_saleprice')


    # data cleansing
    # missing data
    #full_null = full.isnull().sum()
    #full_null[full_null>0].sort_values(ascending=False)

    # fill my median
    full["LotFrontage"] = full.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    # fill missing value
    cols=["MasVnrArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "GarageArea"]
    for col in cols:
        full[col].fillna(0, inplace=True)

    cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish",
             "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
    for col in cols1:
        full[col].fillna("None", inplace=True)

    cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
    for col in cols2:
        full[col].fillna(full[col].mode()[0], inplace=True)

    # full.isnull().sum()[full.isnull().sum()>0]

    # transform numeric features into categorical features
    NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
    for col in NumStr:
        full[col]=full[col].astype(str)


    # create one important feature which is the total area of basement, first and second floor areas of each house
    full['TotalSF'] = full['TotalBsmtSF'] + full['1stFlrSF'] + full['2ndFlrSF']

    # drop saleprice
    full.drop(['SalePrice'],axis=1,inplace=True)


    # apply log1p to the skewed features
    for feature in full:
        if full[feature].dtype != "object":
            full[feature] = np.log1p(full[feature])

    # encode categorical features by LabelEncoder or dummies
    # do label encoding for categorical features
    categorical_features = \
        ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
         'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
         'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
         'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallQual',
         'OverallCond', 'YrSold', 'MoSold')
    for categories in categorical_features:
        lbl = LabelEncoder()
        lbl.fit(list(full[categories].values))
        full[categories] = lbl.transform(list(full[categories].values))
    print('[data_processing] ', 'Shape all_data: {}'.format(full.shape))


    # get dummy categorical features
    full = pd.get_dummies(full)
    print('[data_processing] ', full.shape)

    # save the original data for later use
    full2 = full.copy()

    # train and test dataset
    n_train=train.shape[0]
    X = full2[:n_train]
    test_X = full2[n_train:]
    y = train.SalePrice
    y_log = np.log(train.SalePrice)

    # use RobustScaler because there are some outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit(X).transform(X)
    test_X_scaled = scaler.transform(test_X)

    # feature importance
    lasso=Lasso(alpha=0.001)
    lasso.fit(X,y_log)
    lasso_feat_import = pd.DataFrame({"Feature Importance":lasso.coef_}, index=full2.columns)
    lasso_feat_import.sort_values("Feature Importance",ascending=False)

    # use PCA to address multicollinearity
    pca = PCA(n_components=250)
    X_scaled=pca.fit_transform(X_scaled)
    test_X_scaled = pca.transform(test_X_scaled)

    # X_scaled.shape, test_X_scaled.shape

    # define cross validation strategy
    def rmse_cv(model,X,y):
        rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
        return rmse


    models = [LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),
              ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
              ExtraTreesRegressor()]

    names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb"]
    for name, model in zip(names, models):
        score = rmse_cv(model, X_scaled, y_log)
        print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))

    # must do imputer first, otherwise stacking won't work, and i don't know why.
    a = Imputer().fit_transform(X_scaled)
    b = Imputer().fit_transform(y_log.values.reshape(-1,1)).ravel()

    lasso = Lasso(alpha=0.0005,max_iter=10000)
    ridge = Ridge(alpha=60)
    svr = SVR(gamma= 0.0004,kernel='rbf',C=13,epsilon=0.009)

    ela = ElasticNet(alpha=0.005,l1_ratio=0.08,max_iter=10000)
    bay = BayesianRidge()

    rf = RandomForestRegressor()
    gbr = GradientBoostingRegressor()
    etr = ExtraTreesRegressor()

    ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=0.8)

    stack_model = stacking(mod=[lasso, ridge, svr, ker, ela, bay, rf, gbr, etr],meta_model=ker)

    #print(rmse_cv(stack_model,a,b))
    #print(rmse_cv(stack_model,a,b).mean())

    stack_model.fit(a,b)
    pred = np.exp(stack_model.predict(test_X_scaled))

    result=pd.DataFrame({'Id':test.Id, 'SalePrice':pred})
    result.to_csv("stack_submission.csv",index=False)
    print('stacking model is completed')
    # the best benchmark result on Kaggle is 0.12220


    # deep learning model
    seed = 123
    np.random.seed(seed)

    # Model
    model = Sequential()
    model.add(Dense(500, input_dim=250, kernel_initializer='normal', activation='relu'))
    model.add(Dense(250, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())


    model.fit(X_scaled, y_log.values, epochs=100, batch_size=10)
    model.evaluate(X_scaled, y_log.values)

    dl_pred = np.exp(model.predict(test_X_scaled))
    dl_result = pd.DataFrame({'Id':test.Id, 'SalePrice':dl_pred.reshape(1459,)})
    dl_result.to_csv("dl_submission.csv",index=False)
    print('deep learning model is completed!')


process_data()
print('The end!')
