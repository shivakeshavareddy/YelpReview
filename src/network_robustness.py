# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite
import pandas as pd
import re
import matplotlib
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession, Row
from pyspark.sql import SQLContext

user_reg = 'user'
user_elite = 'elite_user'
business = 'biz'

spark = SparkSession.builder.appName('Yelp_DS_Analysis').config("spark.executor.memory", '32g').config(
    "spark.driver.memory", '32g').config('spark.executor.cores', '6').config('spark.driver.maxResultSize',
                                                                             '6g').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

# %%%%%%%
user_data = spark.read.json('user.json')
business_data = spark.read.json('business.json')
checkin_data = spark.read.json('checkin.json')
review_data = spark.read.json('review.json')
tips_data = spark.read.json('tip.json')

# %%%%%%%
df_business_data = business_data.toPandas()
df_checkin_data = checkin_data.toPandas()
df_tips_data = tips_data.toPandas()
df_user_data = user_data.toPandas()
df_review_data = review_data.toPandas()

# %%%%%%%
# sampled_user_data = user_data.sample(False, 0.10, 42)
# df_user_data = sampled_user_data.toPandas()

# sampled_user_data = user_data.sample(False, 0.25, 42)
# =============================================================================
# temp = sampled_user_data.withColumnRenamed('cool', 'cool_userdt') \
#     .withColumnRenamed('useful', 'useful_usredt') \
#     .withColumnRenamed('user_id', 'user_id_userdt') \
#     .withColumnRenamed('funny', 'funny_userdt')
# 
# user_reviews = temp.join(review_data, temp.user_id_userdt == review_data.user_id)
# 
# user_reviews = user_reviews.drop('cool_userdt', 'useful_usredt', 'user_id_userdt', 'funny_userdt', 'average_stars',
#                                  'compliment_cool', 'compliment_cute',
#                                  'compliment_funny', 'compliment_hot', 'compliment_list', 'compliment_more',
#                                  'compliment_note', 'compliment_photos',
#                                  'compliment_plain', 'compliment_profile', 'compliment_writer', 'yelping_since', 'fans',
#                                  'name', 'review_count', 'elite', 'friends')
# 
# user_reviews.head()
# print(sampled_user_data.count())
# print(user_reviews.count())
# print(review_data.count())
# print(user_reviews.schema)
# =============================================================================

# sampled_review_data = review_data.sample(False, 0.25, 42)
# df_review_data = user_reviews.toPandas()
# %%%%%%%
print('Reviews:', len(df_review_data))
print('Users:', len(set(df_review_data.user_id)))
print('Businesses:', len(set(df_review_data.business_id)))


# %%%%%%%
# A node class for storing data.
class Node:
    def __init__(self, Data, Type):
        self.Data = Data
        self.Type = Type

    def to_string(self):
        return "Node (%s), Data: " % (self.Type, self.Data)

    def __hash__(self):
        return hash(self.Data)

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.Data == other.Data
        )


# %%%%%%%
# creating spearate list of elite and non elite users
elite_users = df_user_data[df_user_data['elite'] != '']
regular_users = df_user_data[df_user_data['elite'] == '']

# %%%%%%%
# Plotting Friend Degree Distirbution

import collections

non_elite_friend_count = list(regular_users['friends'].str.count(",") + 1)
non_elite_friend_count.sort(reverse=True)

elite_friend_count = list(elite_users['friends'].str.count(",") + 1)
elite_friend_count.sort(reverse=True)

all_friend_count = list(df_user_data['friends'].str.count(",") + 1)
all_friend_count.sort(reverse=True)

# %%%%%%%

deg_count = collections.Counter(non_elite_friend_count)
de_gree, counts = zip(*deg_count.items())

plt.title("Friends Degree distribution - Non-elites")
plt.ylabel("Count")
plt.xlabel("Users")
plt.bar(de_gree, tuple([(x - min(counts)) / (max(counts) - min(counts)) for x in counts]))
plt.xlim(-5, 200)
plt.ylim(0, 1.05)
plt.savefig("FDD_NonElite.png")

# %%%%%%%
deg_count = collections.Counter(elite_friend_count)
de_gree, counts = zip(*deg_count.items())

plt.title("Friends Degree distribution - Elites")

plt.ylabel("Count")
plt.xlabel("Users")
plt.bar(de_gree, tuple([(x - min(counts)) / (max(counts) - min(counts)) for x in counts]))
plt.xlim(-5, 1000)
plt.ylim(0, 1.05)
plt.savefig("FDD_Elites.png")

# %%%%%%%

deg_count = collections.Counter(all_friend_count)
de_gree, counts = zip(*deg_count.items())

deg_count = collections.Counter(all_friend_count)
de_gree, counts = zip(*deg_count.items())

plt.title("Friends Degree distribution - All")

plt.ylabel("Count")
plt.xlabel("Users")
plt.bar(de_gree, tuple([(x - min(counts)) / (max(counts) - min(counts)) for x in counts]))
plt.xlim(-5, 200)
plt.ylim(0, 1.05)
plt.savefig("FDD_All.png")

# %%%%%%%
########################################
# Connected Components
########################################
# Run This - 1

# in elites edges is added non-elites
elite_user_array = []
friend_graph_elite = nx.Graph()
friend_graph_regular = nx.Graph()
friend_graph_all = nx.Graph()

for usr in df_user_data.itertuples(index=True):
    u_id, friends, inf = (usr.user_id, usr.friends, usr.elite)
    if inf != '':
        friend_graph_all.add_node(u_id, elite=True)
        elite_user_array.append(u_id)
    else:
        friend_graph_all.add_node(u_id, elite=False)

# add edges

for usr in df_user_data.itertuples(index=True):
    u_id, friends, inf = (usr.user_id, usr.friends, usr.elite)
    if friends != '':
        for f in friends.split(', '):
            friend_graph_all.add_edge(u_id, f)

len(friend_graph_all.nodes())

len(friend_graph_all.edges())

# %%%%%%%
#################
# remove random from 1 to 100% of elites

import random
import time

quarter_percent_users = int(len(elite_user_array) * 0.25)

random.shuffle(elite_user_array)
print("Elites size: " + str(len(elite_user_array)))
print("1 % is: " + str(quarter_percent_users))

############## elite users
largest_conn_comp = []

for i in range(0, 4):
    start_time = time.time()
    print(str(i) + " percent removed.")
    print("Network size: " + str(len(friend_graph_all)))
    max_conn_component = len(max(nx.connected_component_subgraphs(friend_graph_all), key=len))
    largest_conn_comp.append(max_conn_component)
    print("Largest connected component: " + str(max_conn_component))
    print("--- Time to construct graph %s seconds ---" % (time.time() - start_time))
    for j in range(0, quarter_percent_users):
        # print(j)
        elite_removed = elite_user_array.pop()
        friend_graph_all.remove_node(elite_removed)
    print("--- Complete %s seconds ---" % (time.time() - start_time))

with open('Remove_Elites_LCC.txt', 'w') as f:
    for len_mcc in largest_conn_comp:
        f.write("%s\n" % len_mcc)

# %%%%%%%
############# non elite users
# remove random from 1 to 100% of non-elites
quarter_len = int(len(regular_users) * 0.25)

list_regular_users = list(regular_users.user_id)

random.shuffle(list_regular_users)
print("Non-Elites size: " + str(len(list_regular_users)))
print("25 % is: " + str(quarter_len))

largest_conn_comp_regular = []

for i in range(0, 4):
    print(str(i) + " percent removed.")
    print("Network size: " + str(len(friend_graph_all)))
    max_conn_component = len(max(nx.connected_component_subgraphs(friend_graph_all), key=len))
    largest_conn_comp_regular.append(max_conn_component)
    print("Largest connected component: " + str(max_conn_component))
    for j in range(0, quarter_len):
        regular_removed = list_regular_users.pop()
        friend_graph_all.remove_node(regular_removed)

with open('Remove_Non_Elites_LCC.txt', 'w') as f:
    for len_mcc in largest_conn_comp_regular:
        f.write("%s\n" % len_mcc)

# %%%%%%%
############# all users
# remove random from 1 to 100% of all
# Run This - 2

quarter_len = int(len(df_user_data) * 0.25)

all_users_list = list(df_user_data.user_id)

random.shuffle(all_users_list)
print("User size: " + str(len(all_users_list)))
print("25 % is: " + str(quarter_len))

largest_conn_comp_allusr = []

for i in range(0, 4):
    print(str(i) + " percent removed.")
    print("Network size: " + str(len(friend_graph_all)))
    max_conn_component = len(max(nx.connected_component_subgraphs(friend_graph_all), key=len))
    largest_conn_comp_allusr.append(max_conn_component)
    print("Largest connected component: " + str(max_conn_component))
    for j in range(0, quarter_len):
        removed_all = all_users_list.pop()
        friend_graph_all.remove_node(removed_all)

with open('Remove_All_Users_LCC.txt', 'w') as f:
    for len_mcc in largest_conn_comp_allusr:
        f.write("%s\n" % len_mcc)

# %%%%%%%
##### Remove_Non_Elites_LCC_based_on_elite_size.txt
quarter_len = int(len(elite_user_array) * 0.25)

list_regular_users = list(regular_users.user_id)

largest_conn_comp_elSize = []

random.shuffle(list_regular_users)
print("Non-Elites size: " + str(len(list_regular_users)))
print("1 % is: " + str(quarter_len))

for i in range(0, 4):
    print(str(i) + " percent removed.")
    print("Network size: " + str(len(friend_graph_all)))
    max_conn_component = len(max(nx.connected_component_subgraphs(friend_graph_all), key=len))
    largest_conn_comp_elSize.append(max_conn_component)
    print("Largest connected component: " + str(max_conn_component))
    for j in range(0, quarter_len):
        regular_removed = list_regular_users.pop()
        friend_graph_all.remove_node(regular_removed)

with open('Remove_Reg_Users_LCC.txt', 'w') as f:
    for len_mcc in largest_conn_comp_elSize:
        f.write("%s\n" % len_mcc)

quarter_len = int(len(elite_user_array) * 0.25)

UserListAll = list(df_user_data.user_id)

random.shuffle(UserListAll)
print("Non-Elites size: " + str(len(UserListAll)))
print("1 % is: " + str(quarter_len))

# %%%%%%%
#####  Remove_All_LCC_based_on_elite_size

largest_conn_comp_all_elSize = []

for i in range(0, 4):
    print(str(i) + " percent removed.")
    print("Network size: " + str(len(friend_graph_all)))
    max_conn_component = len(max(nx.connected_component_subgraphs(friend_graph_all), key=len))
    largest_conn_comp_all_elSize.append(max_conn_component)
    print("Largest connected component: " + str(max_conn_component))
    for j in range(0, quarter_len):
        regular_removed = UserListAll.pop()
        friend_graph_all.remove_node(regular_removed)

with open('Remove_All_LCC_based_on_elite_size.txt', 'w') as f:
    for len_mcc in largest_conn_comp_elSize:
        f.write("%s\n" % len_mcc)
