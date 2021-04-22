# import required libraries
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

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
    scores = cross_val_score(clf, X, y, cv=10)
    print(scores)
def logisticRegression(X,y):
    logreg = LogisticRegression(max_iter=999999999, random_state=10)
    cvs = cross_val_score(logreg, X, y, cv=10)
    print(cvs)

if __name__ == '__main__':
    print('accuracy scores of SVR for SUI:')
    SVR(X_SUI,y_SUI)
    print('accuracy scores of SVR for UTI:')
    SVR(X_UTI,y_UTI)
    print('accuracy scores of SVR for OAB:')
    SVR(X_OAB,y_OAB)
    print('accuracy scores of SVR for UUI:')
    SVR(X_UUI,y_UUI)
    print('accuracy score of logistic regression for UUI')
    logisticRegression(X_UUI, y_UUI)
    print("accuracy score of logistic regression for UTI")
    logisticRegression(X_UTI, y_UTI)
    print('accuracy score of logistic regression for OAB')
    logisticRegression(X_OAB, y_OAB)
    print("accuracy score of logistic regression for SUI")
    logisticRegression(X_SUI, y_SUI)

##KNN Classification using accuracy score
def knn(X,y):

    #75 / 25 train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    #feature Min-Max Scaling, if the features vary too much (they do) will mess up the model.
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    #Make sure both training and test set are scaled
    X_test_scaled = scaler.transform(X_test)
    result = cross_val_score(KNeighborsClassifier(n_neighbors = 15),X,y, cv=5)
    print(result)
if __name__ == '__main__':
    print('accuracy scores of KNN:')
    knn(X_UTI,y_UTI)



