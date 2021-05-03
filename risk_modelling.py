# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:25:14 2021

@author: 324910
"""

# Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score

# Load data
ship_data = pd.read_csv('C:/Users/324910/Documents/Projects/Ships/Ships Data.csv',
                       error_bad_lines=False)

status_history = pd.read_csv('C:/Users/324910/Documents/Projects/Ships/Status History.csv',
                       error_bad_lines=False)

# Variables in each
ship_data.info()
status_history.info()

# Missing values
ship_data.loc[:, ship_data.isnull().sum() > 0].isnull().sum().sort_values(ascending=False) ## variables with missing values
status_history.loc[:, status_history.isnull().sum() > 0].isnull().sum().sort_values(ascending=False) ## variables with missing values
#### Only ship data has some missing values. 

# Summary stats
display(ship_data.describe(include=[np.number])) ## df of summary stats for each variable

# Distributions of variables
status_history['Unique Identifier'].value_counts()
status_history['STATUS'].value_counts()
status_history['STATUSDATE'].value_counts()

ship_data['COUNTRY OF REGISTRATION'].value_counts()
ship_data['COUNTRYOFBUILD'].value_counts()
ship_data['CLASS'].value_counts()
ship_data['SHIP TYPE'].value_counts()
ship_data['ACCIDENT DATE'].value_counts()

# Data creation

## Clean date variables
#ship_data['accident_date_clean'] = pd.to_datetime(
#        ship_data['ACCIDENT DATE'], format='%Y-%m-%dT%H:%M:%S.%fZ')
#status_history['status_date_clean'] = pd.to_datetime(
#        status_history['STATUSDATE'], format='%Y%m%d') ## some days are 00 and some months are 00

## status category could be minimised to those four categories
### I wonder if this should be a sampling thing

status_conditions = [
 (       (status_history['STATUS'] == "IN SERVICE/COMMISSION") |
         (status_history['STATUS'] == "U.S. RESERVE FLEET") |  
         (status_history['STATUS'] == "CONVERTING/REBUILDING") |    
         (status_history['STATUS'] == "LAID-UP") |   
         (status_history['STATUS'] == "TO BE BROKEN UP") |
         (status_history['STATUS'] == "IN CASUALTY OR REPAIRING")
         ),
 (       (status_history['STATUS'] == "PROJECTED") |
         (status_history['STATUS'] == "ON ORDER/NOT COMMENCED") |  
         (status_history['STATUS'] == "UNDER CONSTRUCTION") |    
         (status_history['STATUS'] == "KEEL LAID") |   
         (status_history['STATUS'] == "LAUNCHED")
         ),
 (       (status_history['STATUS'] == "HULKED") |
         (status_history['STATUS'] == "SCUTTLED") |  
         (status_history['STATUS'] == "TOTAL LOSS") |
         (status_history['STATUS'] == "BROKEN UP")
         ),
 (       (status_history['STATUS'] == "No Longer Meets IHSF Criteria") |
         (status_history['STATUS'] == "CONTINUED EXISTENCE IN DOUBT") |  
         (status_history['STATUS'] == "UNIDENTIFIED SHIP DATA") |    
         (status_history['STATUS'] == "NEVER EXISTED")
         )
 ]

status_values = ['Live Fleet', 'New Buildings', 'Dead/deletions', 'Other']

status_history['status_category'] = np.select(status_conditions, status_values)
display(status_history.head(10))

## Target variable: If there is an accident date, then = 1. 
ship_data[['accident']] = ship_data[['ACCIDENT DATE']].notnull().astype(int)
ship_data[['accident', 'ACCIDENT DATE']].head(100)

## Joining ship data and status history
ship_and_status_data = pd.merge(ship_data, status_history, how='left', on = ['Unique Identifier'])
merge_check = ship_and_status_data[["Unique Identifier", "status_category"]].dropna()
len(merge_check["Unique Identifier"].unique())/len(ship_data) ## 1% of IDs have matches. Terrible!

# Plots

## Categories against the target variable
def dep_cat_binary_target_freq(df, cat, binary_target, sort = "perc"):
    df = df[[cat,binary_target]].dropna()
    df_new = df.groupby([cat, binary_target]).size().reset_index(name="Frequency")
    df_new['%'] = (df_new['Frequency'] / (df_new.groupby([cat])['Frequency'].transform('sum'))) * 100
    df_new = df_new[df_new[binary_target]==1]
    if sort == "cat":
        df_new = df_new[[cat,'%']].sort_values(cat).set_index(cat)
    else: 
        df_new = df_new[[cat,'%']].set_index(cat).sort_values('%')
    return df_new

dep_cat_binary_target_freq(ship_data, "SHIP TYPE", "accident").plot.barh(figsize=(5, 7))
dep_cat_binary_target_freq(ship_data, "ICE_STRENGTHENED", "accident").plot.barh(figsize=(5, 7))
dep_cat_binary_target_freq(ship_data, "ICE_BREAKING", "accident").plot.barh(figsize=(5, 7))
dep_cat_binary_target_freq(ship_data, "CLASS", "accident").plot.barh(figsize=(5, 15))
dep_cat_binary_target_freq(ship_data, "COUNTRYOFBUILD", "accident").plot.barh(figsize=(5, 15)) # Tells you something about manufacturer
dep_cat_binary_target_freq(ship_data, "COUNTRY OF REGISTRATION", "accident").plot.barh(figsize=(5, 15)) # Tells you something about use of ships?
dep_cat_binary_target_freq(ship_and_status_data, "status_category", "accident").plot.barh(figsize=(5, 15))
dep_cat_binary_target_freq(ship_and_status_data, "STATUS", "accident").plot.barh(figsize=(5, 15))

## Correlation matrix of numerical variables
correlation_table = ship_data.corr() # correlation table
sns.heatmap(ship_data.corr(), annot=True, linewidths =.5); ## heatmap for numerical vars
#### Expect correlation with accident to be weak given the ratio of cases is low.

## Accident by built date
dep_cat_binary_target_freq(ship_data, "BUILTDATE", "accident", sort = "cat").plot.barh(figsize=(5, 15)) # Tells you something about use of ships?

## Distributions of numeric variables
ship_data.hist(figsize=(12, 30), bins=30, grid=False, layout=(8, 3))
sns.despine()
plt.suptitle('Numeric features distribution', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Feature transformation for modelling
ship_data_transformed = ship_data.drop(['ACCIDENT DATE', 'Unique Identifier'], axis = 1)

## One-hot encoding categorical features
categories = ['SHIP TYPE', 'ICE_STRENGTHENED',
              'ICE_BREAKING', 'CLASS', 'COUNTRYOFBUILD',
              'COUNTRY OF REGISTRATION', 'BUILTDATE']
ship_data_transformed = pd.get_dummies(ship_data_transformed, columns = categories).dropna()

## log-transformations of skewed numerical features
skewed = ['BREADTH', 'DEADWEIGHT', 'LENGTH', 
          'GROSSTONNAGE', 'LENGTH']
ship_data_transformed[skewed] = ship_data_transformed[skewed].apply(lambda x: np.log(x + 1))

## normalisations to keep numerical features the same scale
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['BREADTH', 'DEADWEIGHT', 'LENGTH', 
          'GROSSTONNAGE', 'LENGTH']

ship_data_transformed[numerical] = scaler.fit_transform(ship_data_transformed[numerical])

## Target variable
target = ship_data_transformed['accident']
ship_data_transformed = ship_data_transformed.drop(['accident'], axis = 1)

## Shuffle and split data
X_train, X_test, y_train, y_test = train_test_split(ship_data_transformed, 
                                                    target, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# Modelling - Categorical regression models. [DTs, Logistic Reg, SVM etc]
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end - start
        
    # Get the predictions on the test set(X_test),
    # then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on training sample
    results['acc_train'] = accuracy_score(y_train, predictions_train)
        
    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute accuracy on training sample
    results['prec_train'] = precision_score(y_train, predictions_train)
        
    # Compute accuracy on test set
    results['prec_test'] = precision_score(y_test, predictions_test)
    
    # Compute accuracy on training sample
    results['rec_train'] = recall_score(y_train, predictions_train)
        
    # Compute accuracy on test set
    results['rec_test'] = recall_score(y_test, predictions_test)
    
    # Compute F-score on training samples
    results['f_train'] = fbeta_score(y_train, predictions_train, 0.5)
        
    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression, RidgeCV, LassoCV, SGDClassifier

# Initialize the three models
clf_A = DecisionTreeClassifier(random_state = 42)
clf_B = AdaBoostClassifier(random_state = 42)
clf_C = RidgeClassifier(random_state = 42)
clf_D = LogisticRegression(random_state = 42)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_100 = len(X_train)
samples_10 = int(len(X_train)/10)
samples_1 = int(len(X_train)/100)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C, clf_D]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)


results = ['train_time', 'pred_time', 'acc_train', 'acc_test', 'prec_train', 'prec_test','rec_train','rec_test','f_train','f_test']
results = pd.DataFrame(pd.Series(results, name='Metrics')).set_index('Metrics')
for clf in [clf_A, clf_B, clf_C, clf_D]:
    clf_name = clf.__class__.__name__
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        s = pd.Series(train_predict(clf, samples, X_train, y_train, X_test, y_test), name = (clf_name+str(i))).reset_index()
        s = s.rename({'index':'Metrics'}, axis='columns')
        results = pd.merge(results, s, how='left', on = ['Metrics'])


# Extract feature importance from best model
model = clf_B.fit(X_train, y_train)

## Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_

table_importances=pd.DataFrame({"Name":X_train.columns,
                 "Feature Importance":importances})
    
table_importances = table_importances.nlargest(10, 'Feature Importance')

# Extract coefficients from linear model
model_for_inference = clf_C.fit(X_train, y_train)

## Extract the feature importances using .feature_importances_ 
inference_coef = pd.DataFrame(
        {"Name" : X_train.columns, 
         "Coefficient" : model_for_inference.coef_.T[:, 0]}
        ) ## recall is quite weak. But precision is good.
    


import statsmodels.api as sm
sm_model = sm.OLS(y_train, X_train).fit()
sm_predictions = sm_model.predict(X_train)
sm_model.summary()


# My thoughts on this so far:

## The final model is fairly weak I guess.
## Not sure how to incorporate dates to the model when the sample is so much weaker.
## Don't understand the problem that well. Do you want to understand coefficients?
## Is there too much noise in the model? How do you deal with this?
## My understanding of the problem is that you want to understand how much the dataset predicts an accident correctly?
## Also, want to know which features contribute to the probability of an accident happening.



