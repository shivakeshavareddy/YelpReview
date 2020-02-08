import numpy as np
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import types
#!/usr/bin/env python
from pyspark.sql import SparkSession, Row
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, TimestampType
import sys
import os
import glob
import pandas as pd
#from functions import sample_udf 
from pyspark import SparkConf
from pyspark.sql.functions import udf

from pyspark.ml.clustering import LDA

from pyspark.ml.feature import CountVectorizer , IDF
from pyspark.mllib.linalg import Vector, Vectors

spark = SparkSession.builder.appName('Tweeter').config("spark.executor.memory", '8g').config("spark.driver.memory", '8g').config('spark.executor.cores', '6').config('spark.driver.maxResultSize', '2g').getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)


user_data=spark.read.json('C:/Users/gameb/Downloads/yelp-dataset/yelp_academic_dataset_user.json')

business_data=spark.read.json('C:/Users/gameb/Downloads/yelp-dataset/yelp_academic_dataset_business.json')
checkin_data=spark.read.json('C:/Users/gameb/Downloads/yelp-dataset/yelp_academic_dataset_checkin.json')
review_data=spark.read.json('C:/Users/gameb/Downloads/yelp-dataset/yelp_academic_dataset_review.json')
tips_data=spark.read.json('C:/Users/gameb/Downloads/yelp-dataset/yelp_academic_dataset_tip.json')
#%%%%%%%%%%%%%%
#sampled_user_data = user_data.sample(False, 0.50, 42)
#%%%%%%%%%%%%%%
user_df=user_data.toPandas()
user_df['number of Friends'] = user_df['friends'].apply( lambda x : len((x)))
user_df['target']=user_df['elite'].apply(lambda x: x!='')
    
import sklearn as sk

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn import metrics

features=user_df.select_dtypes(include=np.number)

target = user_df['target']
clf_random = RandomForestClassifier(class_weight='balanced')
numFolds=10

kf = KFold(n_splits=numFolds, shuffle=True)

for train_index, test_index in kf.split(features):
    xTrain, xTest =  features.iloc[train_index], features.iloc[test_index]
    yTrain, yTest = target.iloc[train_index], target.iloc[test_index]
    print ('Epoch Score:',clf_random.fit(xTrain,yTrain).score(xTest,yTest))


eliteUsers = user_df[user_df['target']==True]
elitefeatures = eliteUsers.select_dtypes(include=[np.number])

target = eliteUsers['target']
print (clf_random.predict(elitefeatures))
clf_random.score(elitefeatures,target)

print(clf_random.feature_importances_)
import matplotlib.pyplot as plt
importances = clf_random.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_random.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(features.shape[1]):
    print("%d.  %s (%f)" % (f + 1, features.columns[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.bar(range(features.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(features.shape[1]), indices)
plt.xlim([-1, features.shape[1]])
plt.show()

    
    
    # Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.bar(range(features.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(features.shape[1]), indices)
plt.xlim([-1, features.shape[1]])
plt.show()
5
factors = dict()
for f in range(features.shape[1]):
    factors[features.columns[indices[f]]] = importances[indices[f]]

values_list=[]
#define a function to calculate the score from the important features. The score a metric to see 'how elite' a user is.
def getEliteScore(x):
    for z in features.columns:
        values_list.append(factors[z])
    sum=0
    for i in range(len(x)):
        sum+=x[i]*values_list[i]
#    print(sum)
    return sum

#fill missing values with 0.
user_df.fillna(0,inplace=True)


user_df['score']=features.apply(lambda x : getEliteScore(x),axis=1)

#userDf.to_csv('updated_users.csv')
#data frame of elite users
eliteUsers = user_df[user_df['target']==True]
#data fram of normal users
normUsers = user_df[user_df['target']==False]
print ("Elite user stats")
print (eliteUsers['score'].describe())

print ("non-Elite user stats")
print (normUsers['score'].describe())

print ("All user stats")
print (user_df['score'].describe())
#eliteUsers.sort_values('score',ascending=True)

#Create uneven bins to help visualize histogram
bins = [0,1,2,3,4,5,10,20,30,50,100,200,300,400,500,1000]
#smaller scale for normal users
bins2 = [0,1,2,3,4,5,6,7,8,9,10,20]

plt.figure()
plt.title('Elite Users')
plt.xlabel("Score")
plt.ylabel("User Count")
eliteUsers['score'].hist(bins=20)
           
plt.figure()
plt.title('non-Elite Users')
plt.xlabel("Score")
plt.ylabel("User Count")
normUsers['score'].hist(bins=20)
           
plt.figure()
plt.title('non-Elite Users Zoomed in')
plt.xlabel("Score")
plt.ylabel("User Count")
normUsers['score'].hist(bins=bins2)
           
plt.figure()
plt.title('All Users')
plt.xlabel("Score")
plt.ylabel("User Count")
user_df['score'].hist(bins=bins)

eliteScores = eliteUsers[['score']]

#print eliteScores
print (eliteScores['score'].mean())
print (eliteScores['score'].median())
print (eliteScores['score'].max())
print (eliteScores['score'].min())
std = eliteScores['score'].std()
print ('Std. Deviation: ',std)
print ('Outlier Cuttoff: ',std*3)



#This forces the outliers to equal 3 times the std. deviation. (around 1200)
eliteScores['score'].clip(0,3*std)


#Standardize scores to [0,1]
eliteScores['stdScore'] = eliteScores['score'].apply(lambda x : ( x - eliteScores['score'].min() ) / ( eliteScores['score'].max()-eliteScores['score'].min()))
#normalize scores. potentially creates negative values.
eliteScores['normScore'] = eliteScores['score'].apply(lambda x : ( x - eliteScores['score'].mean() ) / ( eliteScores['score'].max()-eliteScores['score'].min()))
5
#the log function adds 1 to the score before taking the log because log(0) = undefined
def adjLog(x):
    return np.log(x+1)

eliteScores['logScore'] =  eliteScores['score'].apply(adjLog)
user_df['logScore'] = user_df['score'].apply(adjLog)
print (eliteScores['stdScore'].describe())
print (eliteScores['normScore'].describe())
print (eliteScores['logScore'].describe())
print (user_df['logScore'].describe())

plt.figure()
plt.title('Standardized Elite User Historgram')
plt.xlabel("Standardized Score")
plt.ylabel("User Count")
eliteScores['stdScore'].hist(bins=20,normed=True)

plt.figure()
plt.title('Normalized Elite User Historgram')
plt.xlabel("Normalized Score")
plt.ylabel("User Count")
eliteScores['normScore'].hist(bins=20,normed=True)

plt.figure()
plt.title('Elite User Log Score')
plt.xlabel("Log Score")
plt.ylabel("User Count")           
eliteScores['logScore'].hist(bins=20)

plt.figure()
plt.title('All Users Log Score')
plt.xlabel("Log Score")
plt.ylabel("User Count")            
user_df['logScore'].hist(bins=20)

import scipy.stats as stats
from scipy.stats import skewnorm
import matplotlib.mlab as mlab

#print eliteScores['logScore']
print (stats.normaltest(eliteScores['logScore']))
print (stats.skewtest(eliteScores['logScore']))

eliteScores['logScore'].hist(bins=50,normed=1)
mu = eliteScores['logScore'].mean()
sigma = eliteScores['logScore'].std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma)
plt.plot(x, p, 'r', linewidth=2)
plt.title('Elite User Log Scores')
plt.xlabel("Log Score")
plt.ylabel("Frequency")

#Standardize log score
eliteScores['logstdScore'] = eliteScores['logScore'].apply(lambda x : ( x - eliteScores['logScore'].min() ) / ( eliteScores['logScore'].max()-eliteScores['logScore'].min()))

#Normalize log score
eliteScores['lognormScore'] = eliteScores['logScore'].apply(lambda x : ( x - eliteScores['logScore'].mean() ) / ( eliteScores['logScore'].max()-eliteScores['logScore'].min()))

print (eliteScores['logstdScore'].describe())
print (eliteScores['lognormScore'].describe())

plt.figure()
eliteScores['logstdScore'].hist(bins=50,normed=1)
mu = eliteScores['logstdScore'].mean()
sigma = eliteScores['logstdScore'].std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma)
plt.plot(x, p, 'r', linewidth=2)
plt.title('Elite User Log Scores')
plt.xlabel("Log Score")
plt.ylabel("Nomalized Frequency")  

plt.figure()
eliteScores['lognormScore'].hist(bins=50,normed=1)
mu = eliteScores['lognormScore'].mean()
sigma = eliteScores['lognormScore'].std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma)
plt.plot(x, p, 'r', linewidth=2)

user_df['probNormalUser'] =  clf_random.predict_proba(features)[:,0]
user_df['probEliteUser'] = clf_random.predict_proba(features)[:,1]
prob_elite=user_df['probEliteUser']>0.80
prob_df=user_df[prob_elite]
prob_df_1=prob_df['probEliteUser']!=1
prob_df=prob_df[prob_df_1]
plt.figure()
plt.title('Probable Users Log Score')
plt.xlabel("Log Score")
plt.ylabel("User Count")            
prob_df['logScore'].hist(bins=20)


