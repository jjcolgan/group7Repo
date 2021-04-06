# import required libraries
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('transposedCo-occurenceDataWithOutcomes.csv')
features = df.columns.drop(['PatientID','outcome']).values

''' create separate dataframes for control & each symptom 
    declare X and y values for each dataframe 
    y = target values; X = rest of data '''

# control & UTI+
UTI = df[df.iloc[:,0].str.contains("^(?:C|E)")]
X_UTI = UTI.drop(['PatientID','outcome'], axis=1).values
y_UTI = UTI['outcome'].values

# control & overactive bladder
OAB = df[df.iloc[:,0].str.contains("^(?:C|OAB)")]
X_OAB = OAB.drop(['PatientID','outcome'], axis=1).values
y_OAB = OAB['outcome'].values

# control & urge urinary incontinence
UUI = df[df.iloc[:,0].str.contains("^(?:C|UUI)")]
X_UUI = UUI.drop(['PatientID','outcome'], axis=1).values
y_UUI = UUI['outcome'].values

# control & stress urinary incontinence
SUI = df[df.iloc[:,0].str.contains("^(?:C|SUI)")]
X_SUI = SUI.drop(['PatientID','outcome'], axis=1).values
y_SUI = SUI['outcome'].values

''' run random forest classification using accuracy score as a measure '''
# 5 folds
def random_forest_before_selection(X,y):
    result = cross_val_score(RandomForestClassifier(n_estimators=1000,random_state=1),X,y, cv=5)
    print(result)

''' feature selection uses test_train_split instead of cross validation, since cross val
    uses the test data in each fold of the cross-validation procedure which was also used 
    to choose the features and this is what biases the performance analysis.'''

def random_forest_feature_select(X,y):
    # Split the data into 20% test and 80% training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # create random forest classifier
    model = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)

    # fit model to train classifier
    model.fit(X_train, y_train)

    # return the bacteria name and its gini importance
    feat_importance = list(zip(features, model.feature_importances_))
    for i in feat_importance:
        print(i)

    # threshold = smallest of top 10 gini importances
    top_10 = (sorted(feat_importance, key=lambda t: t[1], reverse=True)[:10])
    thres = top_10[-1][1]

    # select features with gini importance > threshold
    sel_feat = SelectFromModel(model, threshold=thres)

    # train on selected features
    sel_feat.fit(X_train, y_train)

    # print most important features
    for feature_list_index in sel_feat.get_support(indices=True):
        print(features[feature_list_index])

    # create new dataset with important features & run new cross validation
    X_important_train = sel_feat.transform(X_train)
    print(cross_val_score(RandomForestClassifier(n_estimators=60,random_state=1, n_jobs=-1),X_important_train,y_train, cv=5))

if __name__ == '__main__':
    print('accuracy scores of random forest before feature selection:')
    random_forest_before_selection(X_SUI,y_SUI)
    print('accuracy scores of random forest after feature selection:')
    random_forest_feature_select(X_SUI,y_SUI)

def SVR(X,y):
    clf = svm.SVC(C=1, random_state=10)
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)



if __name__ == '__main__':
    print('accuracy scores of SVR:')
    SVR(X_SUI,y_SUI)

##KNN
def knn(X,y):
    # Grid search code for fiding optimal number of neighbors
    #taken from: https://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
    # instantiate model
    knn = KNeighborsClassifier(n_neighbors=5)
    k_range = list(range(1, 31))
    # create a parameter grid: map the parameter names to the values that should be searched
    # simply a python dictionary
    # key: parameter name
    # value: list of values that should be searched for that parameter
    # single key-value pair for param_grid
    param_grid = dict(n_neighbors=k_range)
    # instantiate the grid
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)
    n=0
    #not the best way of doing this, extracting the value of the best params from the dict grid.best_params_
    for key,value in grid.best_params_.items():
        n=value

    result = cross_val_score(KNeighborsClassifier(n_neighbors = 5),X,y, cv=5)
    print(result)

if __name__ == '__main__':
    knn(X_UTI,y_UTI)
    knn(X_OAB,y_OAB)
    knn(X_UUI,y_UUI)
