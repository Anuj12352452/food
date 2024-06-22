import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('termdeposit_train.csv')
test = pd.read_csv('termdeposit_test.csv')

train.shape, test.shape


train.duplicated().sum(), test.duplicated().sum()

train.isnull().sum()

df = train.copy()
train = train.drop(['duration'], axis=1)

dups = df.duplicated()
print('before are there any duplicates : ', dups.any())
df.drop_duplicates(inplace=True)
# reset indices after dropping rows
df=df.reset_index(drop=True)
print('after are there any duplicates : ', df.duplicated().any())

import scipy.stats as stats
import matplotlib.pyplot as plt
cols = ['age', 'campaign', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m','nr_employed']
for col in cols: 
    fig, axes = plt.subplots(nrows=1,ncols=3, figsize=(15, 4))
    fig.suptitle(col)
    axes[0].boxplot(df[col])
    axes[1].hist(df[col])
    stats.probplot(df[col], dist='norm', plot=axes[2])
    plt.show()
    
    fig, axes = plt.subplots(1,2)
df2 = df
col='campaign'
print("Before Shape:",df2.shape)
axes[0].title.set_text("Before")
sns.boxplot(df2[col],orient='v',ax=axes[0])
# Removing campaign above 50 
df2 = df2[ (df2[col]<50)]
print("After Shape:",df2.shape)
axes[1].title.set_text("After")
sns.boxplot(df2[col],orient='v',ax=axes[1])
df=df2;
plt.show()
# reset indices after dropping rows
df=df.reset_index(drop=True)

fig, axes = plt.subplots(1,2)
plt.tight_layout(0.2)
df2 = df
col='cons_conf_idx'
print("Before Shape:",df2.shape)
axes[0].title.set_text("Before")
sns.boxplot(df2[col],orient='v',ax=axes[0])
# Removing cons_price_idx above -28 
df2 = df2[ (df2[col]<-28)]
print("After Shape:",df2.shape)
axes[1].title.set_text("After")
sns.boxplot(df2[col],orient='v',ax=axes[1])
df=df2;
plt.show()
# reset indices after dropping rows
df=df.reset_index(drop=True)

df['contact'] = df['contact'].astype('category').cat.codes

from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse=False)
columns = ['job','marital','education','default','housing','loan','month','day_of_week','poutcome']
df_encoded = pd.DataFrame (encoder.fit_transform(df[columns]))
df_encoded.columns = encoder.get_feature_names(columns)
df.drop(columns ,axis=1, inplace=True)
df= pd.concat([df, df_encoded ], axis=1)

#Do the logarithm trasnformations for required features
from sklearn.preprocessing import FunctionTransformer
logarithm_transformer = FunctionTransformer(np.log1p, validate=True)
# apply the transformation to your data
columns = ['age', 'campaign', 'previous']
to_right_skewed = logarithm_transformer.transform(df[columns])
df['age'] = to_right_skewed[:, 0]
df['campaign'] = to_right_skewed[:, 1]
df['previous'] = to_right_skewed[:, 2]

columns = ['nr_employed']
exp_transformer = FunctionTransformer(lambda x:x**2, validate=True) # FunctionTransformer(np.exp, validate=True) #
to_left_skewed = exp_transformer.transform(df[columns])
df['nr_employed'] = to_left_skewed[:, 0]

from sklearn.preprocessing import KBinsDiscretizer
data = pd.DataFrame(df, columns=['age'])
# fit the scaler to the  data
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans') 
discretizer.fit(data)
_discretize = discretizer.transform(data)
x = pd.DataFrame(_discretize, columns=['age'])
df['age'] = x['age']

from sklearn.preprocessing import StandardScaler
df2 = df
# Removing Categorical Features before the feature scaling
columns = df.columns
# Continous col
columns_cont = np.delete(columns,np.s_[9:])
# Categorical col
columns_categorical = np.delete(columns,np.s_[0:9])
# except age since it is discretized
except_age_cont = np.delete(columns_cont, [0])
# Applying Standardization 
# Init StandardScaler
scaler = StandardScaler()
#Transformation of training dataset features
Except = pd.DataFrame(df, columns = except_age_cont)
scaler.fit(Except)
df = pd.DataFrame(scaler.transform(Except), columns = except_age_cont).join(df[columns_categorical])
df = df.join(df2['age'])
# Get age in last column to first column
cols = list(df.columns)
cols = [cols[-1]] + cols[:-1]  #make last column first
df=df[cols]

from sklearn.model_selection import train_test_split
# set apparent temperature as target
columns_value = df.columns
index = np.argwhere(columns_value == 'y')
columns_value_new = np.delete(columns_value, index)
data = pd.DataFrame(df, columns=columns_value_new)
# target as Y
selected_columns = ['y']
y_true = df[selected_columns].copy()
# X as indipendent 
X = data
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# drop emp.var.rate, cons.price.idx ,euribor3m
X_train = X_train.drop('emp_var_rate', 1)
X_train = X_train.drop('cons_price_idx', 1)
X_train = X_train.drop('euribor3m', 1)
X_test = X_test.drop('emp_var_rate', 1)
X_test = X_test.drop('cons_price_idx', 1)
X_test = X_test.drop('euribor3m', 1)

from sklearn.decomposition import PCA
# see explained variance ratios
pca = PCA()
pca.fit(X_train)
pca.explained_variance_ratio_

pca.explained_variance_ratio_[:25].sum

pca = PCA(n_components = 25)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

 #Import svm model
from sklearn import svm
#Create a svm Classifier
#gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. Higher the value of gamma, will try to exact fit the as per training data set i.e. generalization error and cause over-fitting problem.
clf = svm.SVC(kernel='rbf', gamma=1)
#Train the model using the training sets
clf.fit(X_train_pca, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test_pca)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize = (8,8))
fig, ax = plt.subplots(1)
ax = sns.heatmap(cm, ax=ax, annot=True) #normalize='all'
plt.title('Confusion matrix')
plt.ylabel('True category')
plt.xlabel('Predicted category')
plt.show()

print("Precision : ", metrics.precision_score(y_test, y_pred))
print("Recall : ", metrics.recall_score(y_test, y_pred))
print("f1_score:", metrics.f1_score(y_test, y_pred, average="macro"))

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state = 100)
smt = SMOTE(random_state = 101)
X_train, y_train = smt.fit_resample(X_train, y_train)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)




   


    
