# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 14:36:55 2021

@author: 324910
"""

## Data Info: http://insideairbnb.com/get-the-data.html

# Brief

#	      The firm would like to use a second model to help advise their clients 
#        on how to maximise positive consumer response to their first Airbnb 
#        listing. They are interested in predicting this feature because higher 
#        review scores lead to a higher ranking in Airbnb search results. The 
#        second model must predict an overall customer review metric for a 
#        listing. You are free to define this target metric as you see fit. 
#        You can use any features to build the model but must include the 
#        proposed listing price and the features named description and 
#        neighbourhood_overview. 
#        

## Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from time import time
import nltk
#import visuals as vs

## Load data
listings = pd.read_csv('C:/Users/324910/Documents/Projects/Airbnb/Data/listings_cleaned.csv.gz', 
                       compression='gzip',
                       error_bad_lines=False)

## Thoughts: 
## - Scrape the picture and analyse its relationship with price. Use it to gather descriptive variables.

# Data checking -------------------------------------------------------------------------------

## What variables does this have?
listings.info() # dataframe variables
listings.head(1) # print first row

listings.loc[:, listings.isnull().sum() > 0].isnull().sum().sort_values(ascending=False) ## variables with missing values

summary_stats = listings.describe(include=[np.number]) ## df of summary stats for each variable

## Data cleaning ---------------------------------------------------------------------------------------------------------
## Note: Do not use neighbourhood_group_cleansed or neighbourhood. Use neighbourhood_cleansed instead

## Bathroom variable can be computed from the first number in the bathroom text variable
listings['bathrooms'] = listings['bathrooms_text'].str.extract('^([\\d\\.]+)\\s', expand=False).str.strip().astype(float)
listings = listings.drop(['bathrooms_text'], axis = 1)
## Convert following to 1 and 0: host_is_superhost, host_has_profile_picture, host_identity_verified, has_availability, instant_bookable
##listings.replace({'t': 0, 'f': 1}, inplace=True)

## Remove dollar sign and other characters in price column
listings['price'] = listings['price'].str.replace('([\\$\\,])','').astype(float)

## Count number of amenities
listings['amenities'] = listings['amenities'].str.replace('([\\[\\]\\"])','')
listings['amenities'] = listings.amenities.str.split(pat=",")
listings['amenities_length'] = listings.amenities.map(len)


# Descriptive stats/Distributions on dependent and independent variables -------------------------------------------------

## Distribution of categorical variables 
listings['host_is_superhost'].value_counts() ## Skewed towards central london
listings['neighbourhood_cleansed'].value_counts() ## Skewed towards central london
listings['room_type'].value_counts() ## Mostly private room or entire home
listings['property_type'].value_counts() ## Too many types! And similar to room type. Remove.
listings['amenities'].value_counts() ## Too many types!
listings['instant_bookable'].value_counts() ## Too many types!


listings['neighbourhood_cleansed'].value_counts().sort_values().plot.barh(figsize=(10, 10))
sns.despine()
plt.title('Number of listings by neighbourhood', fontsize=14)

listings['room_type'].value_counts(dropna=False).sort_values().plot.barh()
sns.despine()
plt.title('Number of listings by room type', fontsize=14);

## Distributions of numeric variables
summary_stats = listings.describe(include=[np.number]) ## df of summary stats for each variable

listings.hist(figsize=(12, 30), bins=20, grid=False, layout=(16, 3))
sns.despine()
plt.suptitle('Numeric features distribution', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

listings.price.describe()

# Preparing the data for modelling ---------------------------------------------------------

# FeatureSet for model 1
## - Must use features of the property and listing only

listing_features = listings.drop(['listing_url', 'last_scraped','id', 'calendar_updated', 
                                  'neighbourhood_group_cleansed', 'name', 'description',
                                  'neighborhood_overview', 'picture_url',
                                  'neighbourhood', 'scrape_id', 'number_of_reviews', 
                                  'number_of_reviews_ltm','number_of_reviews_l30d',
                                  'calendar_last_scraped', 'license','first_review', 'last_review',
                                  'minimum_minimum_nights', 'maximum_minimum_nights',
                                  'minimum_maximum_nights', 'maximum_maximum_nights',
                                  'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',
                                  'availability_30', 'availability_60', 'availability_90',
                                  'availability_365','property_type',
                                  'reviews_per_month'], axis = 1)

listing_features = listing_features[
        listing_features.columns[~listing_features.columns.str.contains('host')]
        ] ## remove all host variables

### What about 'amenities'? Ideally add a binary for every amenity but remove for now.
listing_features = listing_features.drop(['amenities'], axis = 1)

listing_features.columns ## columns in final dataset
sns.heatmap(listing_features.corr(), annot=True); ## heatmap for features

listing_features = listing_features.dropna()## dropping nas

## Histogram of features
listing_features.hist(figsize=(12, 30), bins=30, grid=False, layout=(9, 3))
sns.despine()
plt.suptitle('Numeric features distribution', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Log any skewed continuous vars and Normalise numerical features
skewed = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
          'minimum_nights', 'maximum_nights',
           'price',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value',
       'desc_unique', 'neigh_unique', 'amenities_length']
listing_features[skewed] = listing_features[skewed].apply(lambda x: np.log(x + 1))

from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
          'minimum_nights', 'maximum_nights',
           'price',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value',
       'desc_unique', 'neigh_unique', 'amenities_length']

listing_features[numerical] = scaler.fit_transform(listing_features[numerical])

## One-hot encode any categorical features
# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()


categories = ['instant_bookable', 'has_availability',
              'neighbourhood_cleansed', #'property_type', 
              'room_type']
listing_features = pd.get_dummies(listing_features, columns = categories)

# Print the number of features after one-hot encoding
encoded = list(listing_features.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

target_model = listing_features['review_scores_rating'] 
target_model = np.log(target_model + 1)

listing_features = listing_features.drop(['review_scores_rating'], axis = 1)

## Shuffle and split data

from sklearn.cross_validation import train_test_split

#' Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(listing_features, 
                                                    target_model, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

# Modelling ----------------------------------------------------------------------------------------
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

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
            
    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['r2_train'] = r2_score(y_train, predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['r2_test'] = r2_score(y_test, predictions_test)
    
    # Compute F-score on the the first 300 training samples using fbeta_score()
    results['mse_train'] = mean_squared_error(y_train, predictions_train)
        
    # Compute F-score on the test set which is y_test
    results['mse_test'] = mean_squared_error(y_test, predictions_test)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, LassoCV
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor

# Initialize the three models
reg_A = LinearRegression()
reg_B = GradientBoostingRegressor()
reg_C = LinearSVR()
reg_D = RidgeCV()
reg_E = Lasso()
reg_F = KNeighborsRegressor()

# Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_100 = len(X_train)
samples_10 = int(len(X_train)/10)
samples_1 = int(len(X_train)/100)

# Collect results on the learners
results = {}
for clf in [reg_A, reg_B, reg_C, reg_D, reg_E, reg_F]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

#' Train the supervised model on the training set using .fit(X_train, y_train)
model = reg_A.fit(X_train, y_train)

#' Extract the feature importances using .feature_importances_ 
importances = model.coef_

table_importances=pd.DataFrame({"Name":X_train.columns,
                 "Feature Importance":importances})

table_importances = table_importances.nlargest(10, 'Feature Importance')
    
def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = plt.figure(figsize = (12,3.5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    plt.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    plt.xticks(np.arange(5), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)
    
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()  

feature_plot(importances, X_train, y_train)

plt.figure(figsize = (12,3.5))
ax = sns.barplot(x="Feature Importance", y="Name", data=table_importances)
ax.set(xlabel='Relative Importance', ylabel='Feature')
plt.show()
    
# Run metrics visualization for the three supervised learning models chosen
#vs.evaluate(results, accuracy, fscore)
import statsmodels.api as sm
OLS_model_1 = sm.OLS(y_train, X_train).fit()
OLS_predictions_1 = OLS_model_1.predict(X_train)
OLS_model_1.summary()

## ------------------------------------------------------------------------------
## The impact of sub-categories on overal review score -----------------------

import statsmodels.api as sm
# Note the difference in argument order
X = listings[listings.columns[listings.columns.str.contains('^review')]]
X = X.dropna()
y = X['review_scores_rating']
X = X.drop(['review_scores_rating'], axis = 1)

model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary() ## Strongest effects are accuracy, cleanliness, value and communication

