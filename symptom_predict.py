# import required libraries
import pandas as pd
from networkx.drawing.tests.test_pylab import plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from statistics import stdev

# create outfile
from sklearn.preprocessing import StandardScaler

output = open('outfile.txt','w')

# df = pd.read_csv('transposedCo-occurenceDataWithOutcomes.csv')
df = pd.read_csv('new_dataset.csv')
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
# 10 folds
def random_forest_before_selection(X,y):
    # find accuracy score
    result = cross_val_score(RandomForestClassifier(n_estimators=1000,random_state=1),X,y, cv=10)
    output.write('Random forest classification before feature selection - 10 fold (accuracy scores): ' + '\n')
    output.write(str(result) + '\n')
    avg = sum(result)/len(result)
    sdev = stdev(result)
    output.write('Average accuracy score: \n' + str(avg) + '\n')
    output.write('Accuracy score standard deviation: \n' + str(sdev) + '\n')
    output.write('AUC score: \n')

    # find AUC score
    result = cross_val_score(RandomForestClassifier(n_estimators=1000, random_state=1), X, y, scoring='roc_auc', cv=10)
    output.write(str(result) + '\n')
    avg = sum(result) / len(result)
    sdev = stdev(result)
    output.write('average AUC score: \n' + str(avg) + '\n')
    output.write('AUC score standard deviation: \n' + str(sdev) + '\n')

    # find r2 score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # X, y = make_regression(n_features=4, n_informative=2, random_state = 0, shuffle = False)
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    # score = regr.score(y_test,y_pred)
    score = r2_score(y_test,y_pred)
    output.write('R2 score \n')
    output.write(str(score))


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

    ''' output.write('Gini importance of each feature: \n')
    for i in feat_importance:
        output.write(str(i) + '\n') '''

    # threshold = smallest of top 10 gini importances
    top_10 = (sorted(feat_importance, key=lambda t: t[1], reverse=True)[:10])
    thres = top_10[-1][1]

    # select features with gini importance > threshold
    sel_feat = SelectFromModel(model, threshold=thres)

    # train on selected features
    sel_feat.fit(X_train, y_train)

    # print most important features
    output.write('Top 10 most important features: \n')
    for i in sel_feat.get_support(indices=True):
        output.write(str(feat_importance[i]) + '\n')

def elastic_net(X,y):
    # Split the data into 20% test and 80% training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # l1 hyperparameter optimization
    result = ElasticNetCV(l1_ratio=[.05, 0.1, .2, .5, .75, .9, .95, .99, 1], n_jobs=9, tol=0.1)
    # fit data
    result.fit(X_train, y_train)
    # write to output file
    output.write('Elastic net results: \n')
    # predict results
    y_pred = result.predict(X_test)
    # find R^2 - compare predictions to actual values
    score = r2_score(y_test, y_pred)
    output.write('R^2 score: \n' + str(score) + '\n')

def SVC(X,y):
    # accuracy
    clf = svm.SVC(C=1, random_state=10)
    scores = cross_val_score(clf, X, y, cv=10)
    output.write('SVR - 10 fold (accuracy scores): ' + '\n')
    output.write(str(scores) + '\n')
    avg = sum(scores) / len(scores)
    output.write('Average accuracy score: \n' + str(avg) + '\n')

    # AUC
    result = cross_val_score(clf, X, y, scoring='roc_auc', cv=10)
    output.write('AUC scores (10 fold): \n')
    output.write(str(result) + '\n')
    avg = sum(result) / len(result)
    output.write('average AUC score: \n' + str(avg) + '\n')

def logisticRegression(X, y):
    scaler_2 = StandardScaler()
    X_ss = scaler_2.fit_transform(X)
    logreg = LogisticRegression(max_iter=99999999999, random_state=10, C=1)
    scores = cross_val_score(logreg, X_ss, y, cv=10)
    output.write('Logistic Regression - 10 fold (accuracy scores): ' + '\n')
    output.write(str(scores) + '\n')
    avg = sum(scores) / len(scores)
    output.write('Average accuracy score: \n' + str(avg) + '\n')

    result = cross_val_score(logreg, X_ss, y, scoring='roc_auc', cv=10)
    output.write('AUC scores (10 fold): \n')
    output.write(str(result) + '\n')
    avg = sum(result) / len(result)
    output.write('average AUC score: \n' + str(avg) + '\n')

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
# Accuracy scores 
    result = cross_val_score(KNeighborsClassifier(n_neighbors = n),X,y, cv=10)
    output.write('KNN accuracy after features selection (10 fold): \n')
    output.write(str(result) + '\n')
    output.write('average accuracy score: \n')
    avg = sum(result)/len(result)
    output.write(str(avg) + '\n')

#AUC 
    result = cross_val_score(KNeighborsClassifier(n_neighbors = n),X,y, cv=10,scoring='roc_auc')
    output.write(' KNN AUC: \n')
    output.write(str(result) + '\n')
    output.write('Average AUC score: \n')
    avg = sum(result) / len(result)
    output.write(str(avg) + '\n')


# Box plots
def box_plot(X,y,result):
    # accuracy boxplot
    data_to_plot = [result]
    # Create a figure instance
    fig = plt.figure('', figsize=(9, 6))
    # Create an axes instance
    plt.xlabel('Random Forest', fontsize=18)
    ax = fig.add_subplot(111)
    # Create the boxplot
    bp = ax.boxplot(data_to_plot, patch_artist=True)
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='#1b9e77')

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    # Save the figure
    fig.savefig('uui_random_forest.png', bbox_inches='tight')

def main():
    output.write('----------------UTI----------------\n')
    random_forest_before_selection(X_UTI,y_UTI)
    random_forest_feature_select(X_UTI,y_UTI)
    output.write('\n')

    output.write('\n----------------OAB----------------\n')
    random_forest_before_selection(X_OAB,y_OAB)
    random_forest_feature_select(X_OAB,y_OAB)
    output.write('\n')

    output.write('\n----------------UUI----------------\n')
    random_forest_before_selection(X_UUI,y_UUI)
    random_forest_feature_select(X_UUI,y_UUI)
    output.write('\n')

    output.write('\n----------------SUI----------------\n')
    random_forest_before_selection(X_SUI,y_SUI)
    random_forest_feature_select(X_SUI,y_SUI)
    output.write('\n')

    output.write('\n----------------UTI----------------\n')
    elastic_net(X_UTI, y_UTI)
    output.write('\n----------------OAB----------------\n')
    elastic_net(X_OAB, y_OAB)
    output.write('\n----------------UUI----------------\n')
    elastic_net(X_UUI, y_UUI)
    output.write('\n----------------SUI----------------\n')
    elastic_net(X_SUI, y_SUI)

    print('accuracy scores of SVR:')
    SVC(X_SUI,y_SUI)

    output.write('\n----------------UTI----------------\n')
    knn(X_UTI,y_UTI)
    output.write('\n----------------OAB----------------\n')
    knn(X_OAB,y_OAB)
    output.write('\n----------------UUI----------------\n')
    knn(X_UUI,y_UUI)

    output.write('\n----------------SUI----------------\n')
    knn(X_SUI, y_SUI)



if __name__ == '__main__':
    main()


