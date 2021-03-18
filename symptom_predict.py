# import required libraries
import pandas as pd

# read dataframe containing all data
df = pd.read_csv('transposedCo-occurenceDataWithOutcomes.csv')

''' create separate dataframes for control & each symptom 
    declare X and y values for each dataframe 
    y = target values; X = rest of data '''

# control & UTI+
UTI = df[df.iloc[:,0].str.contains("^(?:C|E)")]
X_UTI = UTI.drop(['outcome'], axis=1).values
y_UTI = UTI['outcome'].values

# control & overactive bladder
OAB = df[df.iloc[:,0].str.contains("^(?:C|OAB)")]
X_OAB = OAB.drop(['outcome'], axis=1).values
y_OAB = OAB['outcome'].values

# control & urge urinary incontinence
UUI = df[df.iloc[:,0].str.contains("^(?:C|UUI)")]
X_UUI = UUI.drop(['outcome'], axis=1).values
y_UUI = UUI['outcome'].values

# control & stress urinary incontinence
SUI = df[df.iloc[:,0].str.contains("^(?:C|SUI)")]
X_SUI = SUI.drop(['outcome'], axis=1).values
y_SUI = SUI['outcome'].values

def cross_fold_validation():
    pass

