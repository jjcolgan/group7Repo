# import required libraries
import pandas as pd

# read dataframe containing all data
df = pd.read_csv('transposedCo-occurenceDataWithOutcomes.csv')

''' create separate dataframes for control & each symptom '''

# control & UTI+
UTI = df[df.iloc[:,0].str.contains("^(?:C|E)")]

# control & overactive bladder
OAB = df[df.iloc[:,0].str.contains("^(?:C|OAB)")]

# control & urge urinary incontinence
UUI = df[df.iloc[:,0].str.contains("^(?:C|UUI)")]

# control & stress urinary incontinence
SUI = df[df.iloc[:,0].str.contains("^(?:C|SUI)")]

