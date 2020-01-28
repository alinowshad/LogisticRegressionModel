import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
plt.rc("font", size=14)
names = ['Age', 'Workclass', 'Financial Weight', 'Education', 'Education-num', 'Marital-Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hour-per-week', 'Native-country', 'Class']
missing_values = ["?"]
dataframe = pd.read_csv(r"C:\Users\ali\Desktop\Data1.csv", names=names)
dataframe = dataframe.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
dataframe = dataframe.replace('[?]', np.nan, regex=True)
dataframe['Sex'] = dataframe['Sex'].map({'Female': 0, 'Male':1})
dataframe['Class'] = dataframe['Class'].map({'<=50K': 0, '>50K':1})
dataframe['Workclass'] = dataframe['Workclass'].map({'Federal-gov':1, 'Local-gov':2, 'Private':3, 'Self-emp-inc':4, 'Self-emp-not-inc':5, 'State-gov':6, 'Without-pay':7})
dataframe['Marital-Status'] = dataframe['Marital-Status'].map({'Divorced':1, 'Married-AF-spouse':2, 'Married-civ-spouse':3, 'Married-spouse-absent':4, 'Never-married':5, 'Separated':6, 'Widowed':7})
dataframe['Occupation'] = dataframe['Occupation'].map({'Adm-clerical':1, 'Armed-Forces':2, 'Craft-repair':3, 'Exec-managerial':4, 'Farming-fishing':6, 'Handlers-cleaners':7, 'Machine-op-inspct':8, 'Other-service':9, 'Priv-house-serv':10, 'Prof-specialty':11, 'Protective-serv':12, 'Sales':13, 'Tech-support':14, 'Transport-moving':15})
dataframe['Relationship'] = dataframe['Relationship'].map({'Husband':1, 'Not-in-family':2, 'Other-relative':3, 'Own-child':4, 'Unmarried':5, 'Wife':6})
dataframe['Race'] = dataframe['Race'].map({'Amer-Indian-Eskimo':1, 'Asian-Pac-Islander':2, 'Black':3, 'Other':4, 'White':5})
dataframe['Native-country'] = dataframe['Native-country'].map({'Cambodia':1,'Canada':2, 'China':3, 'Columbia':4,
'Cuba':5, 'Dominican-Republic':6, 'Ecuador':7, 'El-Salvador':8, 'England':9, 'France':10, 'Germany':11, 'Greece':12,
'Guatemala':13, 'Haiti':14, 'Holand-Netherlands':15, 'Honduras':16, 'Hong':17, 'Hungary':18, 'India':19, 'Iran':20,
'Ireland':21, 'Italy':22, 'Jamaica':23, 'Japan':24, 'Laos':25, 'Mexico':26, 'Nicaragua':27, 'Outlying-US(Guam-USVI-etc)':28, 'Peru':29,
'Philippines':30, 'Poland':31, 'Portugal':32, 'Puertoc-Rico':33, 'Scotland':34, 'South':35, 'Taiwan':36, 'Thailand':37, 'Trinadad&Tobago':38, 'United-States':39, 'Vietnam':40, 'Yugoslavia':41})
dataframe = dataframe.drop('Education', 1)
#df = dataframe.groupby('Native-country')
#print(df.first())
dataframe = dataframe.dropna()
array = dataframe.values
X = array[:, 0:13]
Y = array[:,13]
Y = Y.astype('int')
X = X.astype('int')
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression(max_iter=3000)
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
result *= 100
print("Accuracy LogisticRegression Model:" ,result)
model = GaussianNB()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
result *= 100
print("Accuracy Naive Bayes Model:" ,result)
