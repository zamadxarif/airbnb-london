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
from sklearn.cross_validation import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score

# Load data
df_screening = pd.read_csv('C:/Users/324910/Documents/Projects/Autism Screening/Autism-Adult-Data.csv',
                       error_bad_lines=False)

# Variables in each
df_screening.info()

# Missing values
df_screening.loc[:, df_screening.isnull().sum() > 0].isnull().sum().sort_values(ascending=False) ## variables with missing values

# Summary stats
summary_stats = df_screening.describe(include=[np.number]) ## df of summary stats for each variable

# Distributions of variables
df_screening['Age'].value_counts()
df_screening['Gender'].value_counts()
df_screening['Ethnicity'].value_counts()
df_screening['Jundice'].value_counts()
df_screening['Family'].value_counts()
df_screening['Country_of_res'].value_counts() ## could reduce number of categories here
df_screening['Used_app_before'].value_counts()
df_screening['Age_desc'].value_counts() ## discard this
df_screening['Relation'].value_counts()
df_screening['ASD'].value_counts()

# Data creation

## Reducing categories in country of residence
df_screening.loc[
        (df_screening.Country_of_res != "'United States'") & 
        (df_screening.Country_of_res != "'New Zealand'") &
        (df_screening.Country_of_res != "'United Arab Emirates'") & 
        (df_screening.Country_of_res != "India") &
        (df_screening.Country_of_res != "'United Kingdom'") & 
        (df_screening.Country_of_res != "Australia") &
        (df_screening.Country_of_res != "'Sri Lanka'") & 
        (df_screening.Country_of_res != "Afghanistan") & 
        (df_screening.Country_of_res != "France") , 'Country_of_res'] = "Other"


## Target variable: ASD Diagnosis 
df_screening['ASD_target'] = df_screening['ASD'].apply(lambda x: 1 if x == 'YES' else 0)
df_screening[['ASD_target', 'ASD']].head(10)

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

dep_cat_binary_target_freq(df_screening, "Age", "ASD_target", sort = "cat").plot.barh(figsize=(4, 7))
dep_cat_binary_target_freq(df_screening, "Gender", "ASD_target").plot.barh(figsize=(4, 7))
dep_cat_binary_target_freq(df_screening, "Ethnicity", "ASD_target").plot.barh(figsize=(4, 7)) ## Possibly reduce cats here
dep_cat_binary_target_freq(df_screening, "Jundice", "ASD_target").plot.barh(figsize=(3, 7))
dep_cat_binary_target_freq(df_screening, "Family", "ASD_target").plot.barh(figsize=(3, 7)) 
dep_cat_binary_target_freq(df_screening, "Country_of_res", "ASD_target").plot.barh(figsize=(3, 7))
dep_cat_binary_target_freq(df_screening, "Used_app_before", "ASD_target").plot.barh(figsize=(5, 7))
dep_cat_binary_target_freq(df_screening, "Age_desc", "ASD_target").plot.barh(figsize=(5, 7))
dep_cat_binary_target_freq(df_screening, "Relation", "ASD_target").plot.barh(figsize=(5, 7)) 

dep_cat_binary_target_freq(df_screening, "A1_Score", "ASD_target").plot.barh(figsize=(2.5, 2.5))
dep_cat_binary_target_freq(df_screening, "A2_Score", "ASD_target").plot.barh(figsize=(2.5, 2.5))
dep_cat_binary_target_freq(df_screening, "A3_Score", "ASD_target").plot.barh(figsize=(2.5, 2.5))
dep_cat_binary_target_freq(df_screening, "A4_Score", "ASD_target").plot.barh(figsize=(2.5, 2.5))
dep_cat_binary_target_freq(df_screening, "A5_Score", "ASD_target").plot.barh(figsize=(2.5, 2.5))
dep_cat_binary_target_freq(df_screening, "A6_Score", "ASD_target").plot.barh(figsize=(2.5, 2.5))
dep_cat_binary_target_freq(df_screening, "A7_Score", "ASD_target").plot.barh(figsize=(2.5, 2.5))
dep_cat_binary_target_freq(df_screening, "A8_Score", "ASD_target").plot.barh(figsize=(2.5, 2.5))
dep_cat_binary_target_freq(df_screening, "A9_Score", "ASD_target").plot.barh(figsize=(2.5, 2.5))
dep_cat_binary_target_freq(df_screening, "A10_Score", "ASD_target").plot.barh(figsize=(2.5, 2.5))

## Distributions of numeric variables
df_screening.hist(figsize=(12, 30), bins=30, grid=False, layout=(8, 3))
sns.despine()
plt.suptitle('Numeric features distribution', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Feature transformation for modelling
df_screening_transformed = df_screening.drop(['ASD', "Age_desc", "Relation"], axis = 1) ## removing original ASD var 

## One-hot encoding categorical features
categories = ['Gender', 'Ethnicity', 'Jundice', 'Family', 
              'Country_of_res', 'Used_app_before']
df_screening_transformed = pd.get_dummies(df_screening_transformed, columns = categories).dropna()



## Target variable
target = df_screening_transformed['ASD_target']
df_screening_transformed = df_screening_transformed.drop(['ASD_target'], axis = 1)

## Shuffle and split data
X_train, X_test, y_train, y_test = train_test_split(df_screening_transformed, 
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
#from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression

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

plt.figure(figsize = (12,3.5))
ax = sns.barplot(x="Feature Importance", y="Name", data=table_importances)
ax.set(xlabel='Relative Importance', ylabel='Feature')
plt.show() ## Features plot 2


# Extract coefficients from linear model
model_for_inference = clf_C.fit(X_train, y_train)

## Extract the feature importances using .feature_importances_ 
inference_coef = pd.DataFrame(
        {"Name" : X_train.columns, 
         "Coefficient" : model_for_inference.coef_.T[:, 0]}
        ) 
    


import statsmodels.api as sm
sm_model = sm.OLS(y_train, X_train).fit()
sm_predictions = sm_model.predict(X_train)
sm_model.summary()





