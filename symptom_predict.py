# import required libraries
import pandas as pd
from matplotlib import pyplot
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression


from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression

# create outfile
output = open('outfile.txt', 'w')

df = pd.read_csv('new_dataset.csv')
features = df.columns.drop(['PatientID', 'outcome']).values

''' create separate dataframes for control & each symptom 
    declare X and y values for each dataframe 
    y = target values; X = rest of data '''

# control & UTI+
UTI = df[df.iloc[:, 0].str.contains("^(?:C|E)")]
X_UTI = UTI.drop(['PatientID', 'outcome'], axis=1).values
y_UTI = UTI['outcome'].values

# control & overactive bladder
OAB = df[df.iloc[:, 0].str.contains("^(?:C|OAB)")]
X_OAB = OAB.drop(['PatientID', 'outcome'], axis=1).values
y_OAB = OAB['outcome'].values

# control & urge urinary incontinence
UUI = df[df.iloc[:, 0].str.contains("^(?:C|UUI)")]
X_UUI = UUI.drop(['PatientID', 'outcome'], axis=1).values
y_UUI = UUI['outcome'].values

# control & stress urinary incontinence
SUI = df[df.iloc[:, 0].str.contains("^(?:C|SUI)")]
X_SUI = SUI.drop(['PatientID', 'outcome'], axis=1).values
y_SUI = SUI['outcome'].values

''' run random forest classification using accuracy score as a measure '''


# 10 folds
def random_forest_before_selection(X, y):
    # find accuracy score
    result = cross_val_score(RandomForestClassifier(n_estimators=1000, random_state=1), X, y, cv=10)
    output.write('Random forest classification before feature selection - 10 fold (accuracy scores): ' + '\n')
    output.write(str(result) + '\n')
    avg = sum(result) / len(result)
    output.write('Average accuracy score: \n' + str(avg) + '\n')
    output.write('AUC score: \n')
    output.write("%0.2f accuracy rf beforewith a standard deviation of %0.2f" % (result.mean(), result.std()))

    # find AUC score
    result = cross_val_score(RandomForestClassifier(n_estimators=1000, random_state=1), X, y, scoring='roc_auc', cv=10)
    output.write(str(result) + '\n')
    avg = sum(result) / len(result)
    output.write('average AUC score: \n' + str(avg) + '\n')

    result = cross_val_score(RandomForestClassifier(n_estimators=1000, random_state=1), X, y, scoring='r2', cv=10)
    output.write('r2 scores (10 fold): \n')
    output.write(str(result) + '\n')
    avg = sum(result) / len(result)
    output.write('average r2 score: \n' + str(avg) + '\n')


''' feature selection uses test_train_split instead of cross validation, since cross val
    uses the test data in each fold of the cross-validation procedure which was also used 
    to choose the features and this is what biases the performance analysis.'''


def random_forest_feature_select(X, y):
    # Split the data into 20% test and 80% training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # create random forest classifier
    model = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)

    # fit model to train classifier
    model.fit(X_train, y_train)

    # return the bacteria name and its gini importance
    feat_importance = list(zip(features, model.feature_importances_))
    output.write('Gini importance of each feature: \n')
    for i in feat_importance:
        output.write(str(i) + '\n')

    # threshold = smallest of top 10 gini importances
    top_10 = (sorted(feat_importance, key=lambda t: t[1], reverse=True)[:10])
    thres = top_10[-1][1]

    # select features with gini importance > threshold
    sel_feat = SelectFromModel(model, threshold=thres)

    # train on selected features
    sel_feat.fit(X_train, y_train)

    # print most important features
    output.write('Top 10 most important features: \n')
    for feature_list_index in sel_feat.get_support(indices=True):
        output.write(str(features[feature_list_index]) + '\n')

    # create new dataset with important features & run new cross validation
    # accuracy score
    X_important_train = sel_feat.transform(X_train)
    result = (cross_val_score(RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1), X_important_train,
                              y_train, cv=10))
    output.write('Classification accuracy after features selection (10 fold): \n')
    output.write(str(result) + '\n')
    output.write('average accuracy score: \n')
    avg = sum(result) / len(result)
    output.write(str(avg) + '\n')
    output.write("%0.2f accuracy RF after with a standard deviation of %0.2f" % (result.mean(), result.std()))

    # AUC score
    result = (cross_val_score(RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1), X_important_train,
                              y_train, scoring='roc_auc', cv=10))
    output.write('AUC after feature selection (10 fold): \n')
    output.write(str(result) + '\n')
    output.write('Average AUC score: \n')
    avg = sum(result) / len(result)
    output.write(str(avg) + '\n')


def elastic_net(X, y):
    # Split the data into 20% test and 80% training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    result = ElasticNetCV(l1_ratio=[.05, 0.1, .2, .5, .75, .9, .95, .99, 1], n_jobs=9)
    result.fit(X_train, y_train)

    output.write('Elastic net results: \n')
    # return best alpha and best L1
    output.write('Best alpha value: ')
    output.write(str(result.alpha_) + '\n')
    output.write('Best l1-ratio: ')
    output.write(str(result.l1_ratio_) + '\n')

    # predict results
    y_pred = result.predict(X_test)

    # find R^2 - compare predictions to actual values
    score = r2_score(y_test, y_pred)
    output.write('R^2 score: ' + str(score) + '\n')


def SVC(X, y):
    #runs cross validation for SVC with 10 folds
    clf = SVC(C=1, random_state=10)
    scores = cross_val_score(clf, X, y, cv=10)
    #outputs to file
    output.write('SVC - 10 fold (accuracy scores): ' + '\n')
    #outputs scores and average of scores
    output.write(str(scores) + '\n')
    avg = sum(scores) / len(scores)
    output.write('Average accuracy score: \n' + str(avg) + '\n')
    output.write("%0.2f accuracy SVC with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
#finds AUC scores and outputs them in the outfile along with the average AUC score
    result = cross_val_score(clf, X, y, scoring='roc_auc', cv=10)
    output.write('AUC scores (10 fold): \n')
    output.write(str(result) + '\n')
    avg = sum(result) / len(result)
    output.write('average AUC score: \n' + str(avg) + '\n')
#finds r2 score with the average and adds to the outfile
    rscores = cross_val_score(clf, X, y, cv=10, scoring = 'r2')
    output.write('r2 scores (10 fold): \n')
    output.write(str(rscores) + '\n')
    avg = sum(rscores) / len(rscores)
    output.write('average r2 score: \n' + str(avg) + '\n')

def logisticRegression(X, y):
    #scales the models
    scaler_2 = StandardScaler()
    X_ss = scaler_2.fit_transform(X)
    #runs the logistic regression model with cross validation of 10 folds
    logreg = LogisticRegression(max_iter=99999999999, random_state=10, C=1)
    scores = cross_val_score(logreg, X_ss, y, cv=10)
    #output accuracy scores with averages to the outfile
    output.write('Logistic Regression - 10 fold (accuracy scores): ' + '\n')
    output.write(str(scores) + '\n')
    print(scores)
    avg = sum(scores) / len(scores)
    output.write('Average accuracy score: \n' + str(avg) + '\n')
    output.write("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
#finds the AUC score and the average and outputs it to outfile
    result = cross_val_score(logreg, X_ss, y, scoring='roc_auc', cv=10)
    output.write('AUC scores (10 fold): \n')
    output.write(str(result) + '\n')
    avg = sum(result) / len(result)
    output.write('average AUC score: \n' + str(avg) + '\n')
#finds the r2 score and average and adds it to the outfile
    result = cross_val_score(logreg, X_ss, y, scoring='roc_auc', cv=10)
    output.write('r2 scores (10 fold): \n')
    output.write(str(result) + '\n')
    avg = sum(result) / len(result)
    output.write('average r2 score: \n' + str(avg) + '\n')


##KNN Classification using accuracy score
def knn(X, y):
    # 75 / 25 train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # feature Min-Max Scaling, if the features vary too much (they do) will mess up the model.
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Make sure both training and test set are scaled
    X_test_scaled = scaler.transform(X_test)
    result = cross_val_score(KNeighborsClassifier(n_neighbors=15), X, y, cv=5)
    print(result)



if __name__ == '__main__':
    print('accuracy scores of KNN:')


##KNN
def knn(X, y):
    # Grid search code for fiding optimal number of neighbors
    # taken from: https://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
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
    n = 0
    # not the best way of doing this, extracting the value of the best params from the dict grid.best_params_
    for key, value in grid.best_params_.items():
        n = value
    # Accuracy scores
    result = cross_val_score(KNeighborsClassifier(n_neighbors=n), X, y, cv=10)
    output.write('KNN accuracy after features selection (10 fold): \n')
    output.write(str(result) + '\n')
    output.write('average accuracy score: \n')
    avg = sum(result) / len(result)
    output.write(str(avg) + '\n')
    output.write("%0.2f accuracy KNN with a standard deviation of %0.2f" % (result.mean(), result.std()))
    # AUC
    result = cross_val_score(KNeighborsClassifier(n_neighbors=n), X, y, scoring='roc_auc', cv=10)
    output.write(' KNN AUC: \n')
    output.write(str(result) + '\n')
    output.write('Average AUC score: \n')
    avg = sum(result) / len(result)
    output.write(str(avg) + '\n')

    result = cross_val_score(KNeighborsClassifier(n_neighbors=n), X, y, scoring='r2', cv=10)
    output.write('r2 scores (10 fold): \n')
    output.write(str(result) + '\n')
    avg = sum(result) / len(result)
    output.write('average r2 score: \n' + str(avg) + '\n')


def main():

    output.write('----------------UTI----------------\n')
    random_forest_before_selection(X_UTI, y_UTI)
  #  random_forest_feature_select(X_UTI, y_UTI)
    output.write('\n')

    output.write('\n----------------OAB----------------\n')
    random_forest_before_selection(X_OAB, y_OAB)
   # random_forest_feature_select(X_OAB, y_OAB)
    output.write('\n')

    output.write('\n----------------UUI----------------\n')
    random_forest_before_selection(X_UUI, y_UUI)
   # random_forest_feature_select(X_UUI, y_UUI)
    output.write('\n')

    output.write('\n----------------SUI----------------\n')
    random_forest_before_selection(X_SUI, y_SUI)
    #random_forest_feature_select(X_SUI, y_SUI)
    output.write('\n')

    output.write('\n----------------UTI----------------\n')
    elastic_net(X_UTI, y_UTI)
    output.write('\n----------------OAB----------------\n')
    elastic_net(X_OAB, y_OAB)
    output.write('\n----------------UUI----------------\n')
    elastic_net(X_UUI, y_UUI)
    output.write('\n----------------SUI----------------\n')
    elastic_net(X_SUI, y_SUI)

    print('accuracy scores of KNN:')
    output.write('\n----------------UTI----------------\n')
    knn(X_UTI, y_UTI)
    output.write('\n----------------OAB----------------\n')
    knn(X_OAB, y_OAB)
    output.write('\n----------------UUI----------------\n')
    knn(X_UUI, y_UUI)
    output.write('\n----------------SUI----------------\n')
    knn(X_SUI, y_SUI)

    output.write('\n----------------SUI----------------\n')
    SVC(X_SUI, y_SUI)
    output.write('\n----------------UTI----------------\n')
    SVC(X_UTI, y_UTI)
    output.write('\n----------------OAB----------------\n')
    SVC(X_OAB, y_OAB)
    output.write('\n----------------UUI----------------\n')
    SVC(X_UUI, y_UUI)

    output.write('\n----------------UUI----------------\n')
    logisticRegression(X_UUI, y_UUI)
    output.write('\n----------------UTI----------------\n')
    logisticRegression(X_UTI, y_UTI)
    output.write('\n----------------OAB----------------\n')
    logisticRegression(X_OAB, y_OAB)
    output.write('\n----------------SUI----------------\n')
    logisticRegression(X_SUI, y_SUI)


if __name__ == '__main__':
    main()


