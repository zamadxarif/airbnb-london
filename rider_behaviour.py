# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:13:41 2021

@author: Zahid
"""

# Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Data import and initial exploration
# -----------------------------------------------------------------------------    

# Load data
df_riders = pd.read_excel('C:/Users/324910/Documents/Projects/Rider behaviour/ds_economist_rider_rewards.xlsx')

# Variables in each
df_riders.info() ## no missing values

# Summary stats
summary_stats = df_riders[['hours_worked','earnings','orders_delivered']].describe(include=[np.number]) ## df of summary stats for each variable
display(summary_stats)

# Number of zones and riders
len(df_riders['zone_id'].unique()) ## 170 zones
len(df_riders['rider_id'].unique()) ## 1000 riders

# Extract variable for day in the week
df_riders['start_day'] = pd.to_datetime(df_riders['start_time']).dt.day_name()

# Extract variable for hour
df_riders['start_hour'] = pd.to_datetime(df_riders['start_time']).dt.hour

# Distributions
df_riders.hist(figsize=(12, 30), bins=30, grid=False, layout=(8, 3))
sns.despine()
plt.suptitle('Distributions', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Relative frequency by buckets
earnings_bins = [-7, 1, 50, 100, 200, 600]
earnings_names = ['<1', '1-50', '51-100', '101-200', '200+']
df_riders['Earnings buckets'] = pd.cut(df_riders['earnings'], earnings_bins, labels=earnings_names)

hours_bins = [-1, 0.2, 0.4, 0.6, 0.8, 1]
hours_names = ['<0.2', '0.21-0.4', '0.41-0.6', '0.61-0.8', '0.81-1']
df_riders['Hours buckets'] = pd.cut(df_riders['hours_worked'], hours_bins, labels=hours_names)

deliveries_bins = [-1, 2, 4, 6, 13]
deliveries_names = ['<2', '2.1-4', '4.1-6', '6.1+']
df_riders['Deliveries buckets'] = pd.cut(df_riders['orders_delivered'], deliveries_bins, labels=deliveries_names)

df_riders['Earnings buckets'].value_counts(normalize=True).sort_index(
        ).plot.barh(figsize=(2, 2)).set_ylabel('Earnings')

df_riders['Hours buckets'].value_counts(normalize=True).sort_index(
        ).plot.barh(figsize=(2, 2)).set_ylabel('Hours worked')

df_riders['Deliveries buckets'].value_counts(normalize=True).sort_index(
        ).plot.barh(figsize=(2, 2)).set_ylabel('Deliveries')

# Plotting earnings vs hours worked
#### Maybe you can work an entire hour and earn nothing
sns.lmplot(data=df_riders,x='hours_worked', y='earnings',
           size = 3)
plt.xlabel('Hours worked')
plt.ylabel('Earnings')

sns.lmplot(data=df_riders,x="hours_worked", y="orders_delivered",
           size = 3)
plt.xlabel('Hours worked')
plt.ylabel('Orders delivered')

# How do hours worked vary by time?
df_riders['start_time'].value_counts(normalize=True)

sns.lmplot(data=df_riders,x='start_hour', y='hours_worked',
           size = 3)
plt.ylabel('Hours worked')
plt.xlabel('Starting hour')

sns.lmplot(data=df_riders,x='start_day', y='hours_worked',
           size = 3)
plt.xlabel('Hours worked')
plt.ylabel('Day')


# How do orders delivered vary by time?

sns.lmplot(data=df_riders,x='start_hour', y='orders_delivered',
           size = 3)
plt.ylabel('Orders delivered')
plt.xlabel('Starting hour')

sns.lmplot(data=df_riders,x='start_day', y='orders_delivered',
           size = 5)
plt.ylabel('Hours worked')
plt.xlabel('Day')

# -----------------------------------------------------------------------------    
# Back-of-the-envelope estimate of labour supply
# -----------------------------------------------------------------------------  

# Generate wage variable: Earnings / Quantity
### To note: Fee varies by distance to restaurant and service
df_riders['fee'] = np.where(
        df_riders['earnings'] != 0, 
         df_riders['earnings']/df_riders['orders_delivered'], 0)

# Incentive cut off dummy
df_riders['cutoff'] = np.where(
        df_riders['orders_delivered'] < 2, 
        1,
        0)

# Incentive cut off dummy and interaction term
df_riders['cutoff_and_fees'] = df_riders['cutoff']*df_riders['fee']

# Sampling
df_riders_sample = df_riders
df_riders_sample = df_riders_sample[
        ['fee', 
         'zone_id', 'start_hour', 'start_day', 
         'hours_worked', 
         'cutoff', 'cutoff_and_fees',
         'orders_delivered'
         ]]

#df_riders_sample = df_riders_sample[
#        df_riders_sample["orders_delivered"] <= 1]

X = df_riders_sample[
        ['fee', 
         'zone_id', 'start_hour', 'start_day'#,
         , 'cutoff', 'cutoff_and_fees'
         ]]
categories = ['zone_id', 'start_hour', 'start_day']
X = pd.get_dummies(X, columns = categories)
#y = df_riders_sample["orders_delivered"]
y = df_riders_sample["hours_worked"]
#y = y.apply(lambda x: np.log(x + 1)) ## log transformation

# OLS model
import statsmodels.api as sm
X = sm.add_constant(X)
sm_model = sm.OLS(y, 
                  X
                  ).fit()
sm_predictions = sm_model.predict(X)
sm_model.summary()


# Analysis

## Average hours_worked for cut off group
df_riders.groupby('cutoff')['hours_worked'].mean() ##0.63

## Average orders delivered for cut off group
df_riders.groupby('cutoff')['orders_delivered'].mean() ##0.40



# -----------------------------------------------------------------------------    
# Analysis by day
# -----------------------------------------------------------------------------  

# Create dataframe for day based variables
df_riders_day = df_riders
df_riders_day['start_date'] = pd.to_datetime(df_riders_day['start_time']).dt.date
df_riders_day['start_date'].value_counts(normalize=True)
df_riders_day = df_riders.groupby(['rider_id', 'start_date'])['hours_worked', 'earnings', 'orders_delivered'].agg('sum')


df_riders_day.hist(figsize=(12, 30), bins=30, grid=False, layout=(8, 3))
sns.despine()
plt.suptitle('Distributions', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Summary stats
summary_stats_day = df_riders_day[['hours_worked','earnings','orders_delivered']].describe(include=[np.number]) ## df of summary stats for each variable
display(summary_stats_day)

# Relative frequency by buckets
earnings_bins = [-7, 1, 500, 1000, 1500, 3000]
earnings_names = ['<1', '1-500', '501-1000', '1001-1500', '1500+']
df_riders_day['Earnings buckets'] = pd.cut(df_riders_day['earnings'], earnings_bins, labels=earnings_names)

hours_bins = [-1, 0.1, 5, 10, 15, 20]
hours_names = ['0', '0.1-5', '5.1-10', '10.1-15', '15+']
df_riders_day['Hours buckets'] = pd.cut(df_riders_day['hours_worked'], hours_bins, labels=hours_names)

deliveries_bins = [-1, 0.1, 10, 20, 30, 40, 50, 70]
deliveries_names = ['0', '0.1-10', '10.1-20', '20.1-30', '30.1-40', '40.1-50', '50+']
df_riders_day['Deliveries buckets'] = pd.cut(df_riders_day['orders_delivered'], deliveries_bins, labels=deliveries_names)

df_riders_day['Earnings buckets'].value_counts(normalize=True).sort_index(
        ).plot.barh(figsize=(2, 2)).set_ylabel('Earnings')

df_riders_day['Hours buckets'].value_counts(normalize=True).sort_index(
        ).plot.barh(figsize=(2, 2)).set_ylabel('Hours worked')

df_riders_day['Deliveries buckets'].value_counts(normalize=True).sort_index(
        ).plot.barh(figsize=(2, 2)).set_ylabel('Deliveries')

# Plotting earnings vs hours worked
#### Maybe you can work an entire hour and earn nothing
sns.lmplot(data=df_riders_day,x='hours_worked', y='earnings',
           size = 3)
plt.xlabel('Hours worked')
plt.ylabel('Earnings')

sns.lmplot(data=df_riders_day,x="hours_worked", y="orders_delivered",
           size = 3)
plt.xlabel('Hours worked')
plt.ylabel('Orders delivered')
