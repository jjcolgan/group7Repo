# import required libraries
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hpsklearn import HyperoptEstimator
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hyperopt import tpe

# read dataframe containing all data
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('transposedCo-occurenceDataWithOutcomes.csv')

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


