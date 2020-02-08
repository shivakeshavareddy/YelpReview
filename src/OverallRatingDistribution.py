import sys
import os
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession, Row
from pyspark.sql import SQLContext

spark = SparkSession.builder.appName('Yelp_DS_Analysis').config("spark.executor.memory", '8g').config(
    "spark.driver.memory", '8g').config('spark.executor.cores', '4').config('spark.driver.maxResultSize',
                                                                            '2g').getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

user_data = spark.read.json('input/yelp_dataset/user.json')
business_data = spark.read.json('input/yelp_dataset/business.json')
checkin_data = spark.read.json('input/yelp_dataset/checkin.json')
review_data = spark.read.json('input/yelp_dataset/review.json')
tips_data = spark.read.json('input/yelp_dataset/tip.json')

df_business_data = business_data.toPandas()

sampled_user_data = user_data.sample(False, 0.05, 42)
df_user_data = sampled_user_data.toPandas()

temp = sampled_user_data.withColumnRenamed('cool', 'cool_userdt') \
    .withColumnRenamed('useful', 'useful_usredt') \
    .withColumnRenamed('user_id', 'user_id_userdt') \
    .withColumnRenamed('funny', 'funny_userdt')

user_reviews = temp.join(review_data, temp.user_id_userdt == review_data.user_id)

user_reviews = user_reviews.drop('cool_userdt', 'useful_usredt', 'user_id_userdt', 'funny_userdt', 'average_stars',
                                 'compliment_cool', 'compliment_cute',
                                 'compliment_funny', 'compliment_hot', 'compliment_list', 'compliment_more',
                                 'compliment_note', 'compliment_photos',
                                 'compliment_plain', 'compliment_profile', 'compliment_writer', 'yelping_since', 'fans',
                                 'name', 'review_count', 'elite', 'friends')

df_review_data = user_reviews.toPandas()

##############################################
# Plotting Begins
##############################################

import numpy as np
import matplotlib.pyplot as plt

# creating separate list of elite and non elite users
elite_user = df_user_data[df_user_data['elite'] != '']

# Extract ratings for elite, regular, and all users
elite_stars = np.array(df_review_data[df_review_data.user_id.isin(elite_user.user_id)].stars)
regular_stars = np.array(df_review_data[~df_review_data.user_id.isin(elite_user.user_id)].stars)
all_stars = np.array(df_review_data.stars)

# Histogram data for regular users ratings
reg = np.histogram(regular_stars, bins=[1, 2, 3, 4, 5, 6])[0]
reg = reg / sum(reg)

# Histogram data for elite users ratings
elit = np.histogram(elite_stars, bins=[1, 2, 3, 4, 5, 6])[0]
elit = elit / sum(elit)

# Histogram data for all users ratings
all_ = np.histogram(all_stars, bins=[1, 2, 3, 4, 5, 6])[0]
all_ = all_ / sum(all_)

# Plot the histogram data# Plot the histogram data
plt.figure(figsize=[15, 10])
x = np.array([1, 2, 3, 4, 5])
dx = 1 / 12  # x-axis space
plt.bar(x - 2 * dx, height=elit, width=2 * dx, color='red', edgecolor='black')
plt.bar(x, height=reg, width=2 * dx, color='seagreen', edgecolor='black')
plt.bar(x + 2 * dx, height=all_, width=2 * dx, color='gold', edgecolor='black')
plt.xlabel('Review stars')
plt.ylabel('Fraction of users')
plt.legend(['Elite users', 'Regular users', 'All users'])
plt.title('Review distribution for Elite and Regular Yelp users')
plt.savefig('review_dist.svg', format='svg', bbox_inches="tight")
plt.savefig("Overall_Rating_Distribution.png")

plt.figure(figsize=[15, 10])
x = np.array([1, 2, 3, 4, 5])
dx = 1 / 12  # x-axis space
plt.bar(x - 2 * dx, height=elit, width=2 * dx, color='red', edgecolor='black')
plt.bar(x, height=reg, width=2 * dx, color='seagreen', edgecolor='black')
plt.bar(x + 2 * dx, height=all_, width=2 * dx, color='gold', edgecolor='black')
plt.xlabel('Review stars')
plt.ylabel('Fraction of users')
plt.legend(['Elite users', 'Regular users', 'All users'])
plt.title('Review distribution for Elite and Regular Yelp users')
plt.savefig('review_dist.svg', format='svg', bbox_inches="tight")
plt.savefig("Overall_Rating_Distribution.png")

# Elite users are more moderate and peak at 4 stars, where regular users are more critical and over-enthustiastic, i.e. giving 1 star reviews, and 5 star reviews.
