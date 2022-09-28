#!/usr/bin/env python

# 2. Typically, this type of problem is a regression problem since the dependent varaible, suicides/100k pop is numeric. However, we will set it up first as a classification problem by first  setting a
#threshold for the suicides/100k pop variable. For example, if it was higher then a threshold it would count as 1 and then 0 if it fell below the threshold. Then we will use a classification method to
#categorize as either 0 or 1. We will then proceed with a regression analysis where we don't end up converting the dependent variable and leave it numeric as is.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("master.csv")

#Don't need country-year

df = df.drop(columns= ['country-year', 'year'])
print(df.dtypes)


#Check for duplicates

sum(df.duplicated())


#Histogram of Suicide Rate

plt.hist(df["suicides/100k pop"], bins= 50)
plt.xlabel("Suicides/100k pop")
plt.ylabel("Counts")


#Number per Country (Not too much correlations for country)

country= df["country"]
suicides=df["suicides/100k pop"]

plt.plot(country, suicides)
plt.xticks(rotation = 90,fontsize=5)

plt.show()


#Sort Countries (There seems to not be too much of a correlation for suicides and countries)
#We will eliminate this variable

#df.groupby('country').agg({"suicides/100k pop".join,}).reset_index()

group =df.groupby('country')

new_df = group.count()
plt.plot(new_df["suicides/100k pop"])
plt.xticks(rotation = 90,fontsize=7)
plt.xticks(np.arange(0, len(new_df["suicides/100k pop"])+1, 2))
plt.show()


#Check Rates By Sex

male = df[df["sex"] == "male"]

female= df[df["sex"] == "female"]

male_suicide= sum(male["suicides_no"])

female_suicide= sum(female["suicides_no"])

print(f"Number of Male Suicides= {male_suicide}")
print(f"Number of female Suicides= {female_suicide}")

#Plot Male vs females (Seems to be clear indication that males are more likely to commit suicide)

sns.barplot(df["sex"], df["suicides_no"])

#Plot Generation
plt.figure(figsize=(10,10))
sns.barplot(df["generation"], df['suicides_no'])
plt.show()

plt.figure(figsize=(10,10))
sns.barplot(df["generation"], df['suicides/100k pop'])

#Check if there are Null Values and Also Print Correlation Matrix

corr_plot= sns.heatmap(df.corr(), annot= True)

#Drop Columns
print(df.isnull().any())
print()
print(df["HDI for year"].isna().sum())

new_df = df.drop(columns= ["HDI for year", "country", " gdp_for_year ($) "])

# pandas get_dummies function is the one-hot-encoder
def encode_onehot(_df, f):
    _df2 = pd.get_dummies(_df[f], prefix='', prefix_sep='').groupby(level=0, axis=1).max().add_prefix(f+' - ')
    df3 = pd.concat([_df, _df2], axis=1)
    df3 = df3.drop([f], axis=1)
    return df3

# Print nominal variables
for f in list(df.columns.values):
    if df[f].dtype == object:
        print(f)

df_o= encode_onehot(new_df, 'generation')

df_o= encode_onehot(df_o, 'sex')

df_o= encode_onehot(df_o, 'age')

df_o


#Get Threshold

y_mean= np.mean(df["suicides/100k pop"])

y_std= df["suicides/100k pop"].std()

threshold= y_mean + 0.5 *(y_std)


#Set Threshold 

df_o['suicides/100k pop'] = df_o['suicides/100k pop'].apply(lambda x: 1 if x > threshold else 0)

df_o


#Split to x and y
X= df_o.loc[:, df_o.columns != 'suicides/100k pop'].values
y= df_o.loc[:, df_o.columns == 'suicides/100k pop'].values.ravel()

#Create Function To Train and Test Model

def model_train_test(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=None, n_jobs=4)
    
    #Train on training data
    model = rf.fit(X_train, y_train)
    # Test on training data
    y_pred = rf.predict(X_test)
    # Return more proper evaluation metric
    # return f1_score(_y_ts, y_pred, pos_label='recurrence-events', zero_division=0)
    # Return accuracy
    return accuracy_score(y_test, y_pred)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=None)
model_train_test(X_train, X_test, y_train, y_test)

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=None)
    print(model_train_test(X_train, X_test, y_train, y_test))


#Look at Decision Tree 
from sklearn import tree

def tree_train_test(X_train, X_test, y_train, y_test):
    dt = tree.DecisionTreeClassifier()
    #Train on training data
    model = dt.fit(X_train, y_train)
    
    # Test on training data
    y_pred = model.predict(X_test)
    # Return more proper evaluation metric
    # Return accuracy
    return accuracy_score(y_test, y_pred)

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=None)
    print(tree_train_test(X_train, X_test, y_train, y_test))

#Decision Tree More Accurate!


#Time to test regression approach by using numerical dependent variable instead
new_df= encode_onehot(new_df, 'generation')

final_df= encode_onehot(new_df, 'sex')

final_df= encode_onehot(final_df, 'age')

final_df


#Should do regression 
new_x = final_df.loc[:, final_df.columns != 'suicides/100k pop'].values
new_y= final_df.loc[:, final_df.columns == 'suicides/100k pop'].values

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(new_x, new_y, test_size=0.20, random_state=None)

model= LinearRegression()
lr= model.fit(X_train, y_train)

predictions= lr.predict(X_test)
predictions


def lr_train_test(X_train, X_test, y_train, y_test):
    model= LinearRegression()
    #Train on training data
    lr= model.fit(X_train, y_train)
    
    # Test on training data
    predictions= lr.predict(X_test)
    # Return more proper evaluation metric
    # return f1_score(_y_ts, y_pred, pos_label='recurrence-events', zero_division=0)
    # Return accuracy
    return r2_score(y_test, predictions)


#Check R2 score
from sklearn.metrics import r2_score
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(new_x, new_y, test_size=0.20, random_state=None)
    print(lr_train_test(X_train, X_test, y_train, y_test))
