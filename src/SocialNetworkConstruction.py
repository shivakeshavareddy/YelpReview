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
elites = 'elite_user'
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

# =============================================================================
# 
# df_business_data = business_data.toPandas()
# 
# df_checkin_data = checkin_data.toPandas()
# # %%%%%%%%
# df_tips_data = tips_data.toPandas()
# 
# # %%%%%%%%
# sampled_user_data = user_data.sample(False, 0.50, 42)
# df_user_data = sampled_user_data.toPandas()
# 
# # %%%%%%%
# # sampled_user_data = user_data.sample(False, 0.25, 42)
# temp = sampled_user_data.withColumnRenamed('cool', 'cool_userdt') \
#     .withColumnRenamed('useful', 'useful_usredt') \
#     .withColumnRenamed('user_id', 'user_id_userdt') \
#     .withColumnRenamed('funny', 'funny_userdt')
# 
# # %%%%%%%
# user_reviews = temp.join(review_data, temp.user_id_userdt == review_data.user_id)
# # %%%%%%%
# user_reviews = user_reviews.drop('cool_userdt', 'useful_usredt', 'user_id_userdt', 'funny_userdt', 'average_stars',
#                                  'compliment_cool', 'compliment_cute',
#                                  'compliment_funny', 'compliment_hot', 'compliment_list', 'compliment_more',
#                                  'compliment_note', 'compliment_photos',
#                                  'compliment_plain', 'compliment_profile', 'compliment_writer', 'yelping_since', 'fans',
#                                  'name', 'review_count', 'elite', 'friends')
# # %%%%%%%
# user_reviews.head()
# print(sampled_user_data.count())
# print(user_reviews.count())
# print(review_data.count())
# print(user_reviews.schema)
# =============================================================================
# %%%%%%%%
# sampled_review_data = review_data.sample(False, 0.25, 42)
# df_review_data = user_reviews.toPandas()

# %%%%%%%%
############### Creating the social network ################
#### Filtering out Elite Users ######

#elite_users = df_user_data[df_user_data.elite != '']
elite_users = df_user_data[df_user_data['elite'] != '']
regular_users = df_user_data[df_user_data['elite'] == '']


####### Making Friend List #######
elite_users_ids = set(elite_users.user_id)

usrs_have_friend = df_user_data[df_user_data.friends != '']

list_friends = dict()
for index, row in usrs_have_friend.iterrows():
    list_friends[row.user_id] = re.split(r',', row.friends)

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


user_ids = set(df_user_data.user_id)


social_network = nx.Graph()
for uid in usrs_have_friend.user_id:
    a = Node(uid, elites if uid in elite_users_ids else user_reg)
    for fid in list_friends[uid]:
        if fid in user_ids:
            b = Node(fid, elites if fid in elite_users_ids else user_reg)
            social_network.add_edge(a, b)


N, L = len(social_network.nodes()), len(social_network.edges())

print('Nodes:', N)
print('Edges:', L)

# %%%%%%%%
#### Degree Distributions #######
nodes_regular_users = [new_node for new_node in list(social_network.nodes()) if new_node.Type == user_reg]
nodes_elite_users = [nw_node for nw_node in list(social_network.nodes()) if nw_node.Type == elites]
nodes_all_users = list(social_network.nodes())

print('Number of Regular User Nodes:', len(nodes_regular_users))
print('Number of Elite User Nodes:', len(nodes_elite_users))


# %%%%%%%%

def degree(g, nodes=None, as_list=True):
    deg_ = dict(g.degree())
    if nodes:
        deg_ = dict(g.degree(nodes))

    if as_list:
        return list(deg_.values())

    return deg_


def degree_plot(my_graph, nodes=None, filename=None, title=''):
    graph_deg = degree(my_graph, nodes=nodes)
    bins = 100
    if len(nodes) < 100:
        bins = len(nodes)
    hist = np.histogram(graph_deg, bins=bins)
    frequencies, edges = hist[0], hist[1]
    n = frequencies.size
    means = [(edges[i] + edges[i + 1]) / 2 for i in range(n)]

    # SCATTER PLOT
    plt.figure(figsize=[15, 10])
    plt.plot(means, frequencies, ".", markersize=20)
    plt.xlabel("k")
    plt.ylabel("frequency")
    plt.title("Degree distribution for %s" % title)
    if filename: plt.savefig('%s.svg' % filename, format='svg', bbox_inches="tight")
    # plt.show()

    # LOG LOG PLOT
    plt.figure(figsize=[15, 10])
    plt.loglog(means, frequencies, ".", markersize=20)
    plt.xlabel("log(k)")
    plt.ylabel("log(frequency)")
    plt.title("Log-log degree distribution for %s" % title)
    if filename: plt.savefig('log_%s.svg' % filename, format='svg', bbox_inches="tight")
    # plt.show()


# %%%%%%%%%%%%%%%
##### Degree Plots #######
degree_plot(social_network, nodes=nodes_regular_users, title="Regular User Nodes", filename='degree_social_regular')

# %%%%%%%%%%%%%%%
degree_plot(social_network, nodes=nodes_all_users, title="All User Nodes", filename='degree_social_all')

# %%%%%%%%%%%%%%%
degree_plot(social_network, nodes=nodes_elite_users, title="Elite User Nodes", filename='degree_social_elite')

# %%%%%%%%%%%%%%%
print('Nodes:', len(social_network.nodes()))
print('Edges:', len(social_network.edges()))

# %%%%%%%%%%%%%%%
L = max(nx.connected_component_subgraphs(social_network), key=len)
print("Nodes in largest sub component:", len(L.nodes()))
print("Edges in largest sub component:", len(L.edges()))

avg_cluster_index = nx.average_clustering(social_network)
print(avg_cluster_index)

# %%%%%%%%%%%%%%%
##### EIGEN VECTOR CENTRALITY #####
eigen_value = nx.eigenvector_centrality_numpy(social_network)
eigen_elite_user_avg = np.mean([eigen_value[node] for node in eigen_value if node.Type == elites])
eigen_regular_avg = np.mean([eigen_value[node] for node in eigen_value if node.Type == user_reg])
eigen_all_avg = np.mean([eigen_value[node] for node in eigen_value if node.Type == user_reg or node.Type == elites])

print(eigen_elite_user_avg, eigen_regular_avg, eigen_all_avg)

# %%%%%%%%%%%%%%%
deg_central = nx.degree_centrality(social_network)
degree_elite_avg = np.mean([deg_central[node] for node in deg_central if node.Type == elites])
degree_regular_avg = np.mean([deg_central[node] for node in deg_central if node.Type == user_reg])
degree_all_avg = np.mean([deg_central[node] for node in deg_central if node.Type == user_reg or node.Type == elites])

print(degree_elite_avg, degree_regular_avg, degree_all_avg)

# %%%%%%%%%%%%%%%
degree_central_elite = np.array([deg_central[node] for node in nodes_elite_users])
eigen_central_elite = np.array([eigen_value[node] for node in nodes_elite_users])

degree_central_regular = np.array([deg_central[node] for node in nodes_regular_users])
eigen_central_regular = np.array([eigen_value[node] for node in nodes_regular_users])

degree_central_all = np.array([deg_central[node] for node in nodes_all_users])
eigen_central_all = np.array([eigen_value[node] for node in nodes_all_users])

# %%%%%%%%%%%%%%%
plt.figure(figsize=[30, 10])
plt.scatter(degree_central_elite, eigen_central_elite, edgecolor='black')
plt.xlabel('Degree centrality')
plt.ylabel('Eigenvalue centrality')
plt.title('Eigenvalue centrality vs. degree centrality for elite users')
plt.savefig('social_ev_elite.png', format='png', bbox_inches="tight")
plt.show()
# %%%%%%%%%%%%%%%
plt.figure(figsize=[30, 10])
plt.scatter(degree_central_regular, eigen_central_regular, edgecolor='black')
plt.xlabel('Degree centrality')
plt.ylabel('Eigenvalue centrality')
plt.title('Eigenvalue centrality vs. degree centrality for regular users')
plt.savefig('social_ev_reg.png', format='png', bbox_inches="tight")
plt.show()
# %%%%%%%%%%%%%%%
plt.figure(figsize=[30, 10])
plt.scatter(degree_central_all, eigen_central_all, edgecolor='black')
plt.xlabel('Degree centrality')
plt.ylabel('Eigenvalue centrality')
plt.title('Eigenvalue centrality vs. degree centrality for all users')
plt.savefig('social_ev_all.png', format='png', bbox_inches="tight")
plt.show()
# %%%%%%%%%%%%%%%
plt.figure(figsize=[30, 10])
plt.scatter(degree_central_elite, eigen_central_elite, edgecolor='black')
plt.xlabel('Degree centrality')
plt.ylabel('Eigenvalue centrality')
plt.show()
# %%%%%%%%%%%%%%%
plt.figure(figsize=[30, 10])
plt.scatter(degree_central_elite, eigen_central_elite, edgecolor='black')
plt.xlabel('Degree centrality')
plt.ylabel('Eigenvalue centrality')
plt.show()

# %%%%%%%%%%%%%%%
##### Removing Elite Users for Robustness Analysis ####
import random, copy


def robustness_analysis(my_graph, nodes, quarter_percent_users):
    k = 4
    random.shuffle(nodes)

    largest_conn_comp_val = np.zeros(k)

    # Start loop
    for i in range(k):
        print(str(i) + " percent removed.")
        print("Network size: " + str(len(my_graph)))

        # Compute largest connected subcomponent
        mcc = len(max(nx.connected_component_subgraphs(my_graph), key=len))
        largest_conn_comp_val[i] = mcc
        print("Largest connected component: " + str(mcc))

        # Remove 1 percent of users, randomly chosen
        for j in range(quarter_percent_users):
            node = nodes.pop()
            my_graph.remove_node(node)


# one_percent = int(len()*0.01)
# %%%%%%%%%%%%%%%
# one_percent = int(len()*0.01)
large_conn_comp_all = pd.read_csv('rem_all_size.txt', header=None, names=['data'])
large_conn_comp_regular = pd.read_csv('rem_reg_size.txt', header=None, names=['data'])
large_conn_comp_elite = pd.read_csv('rem_elite_size.txt', header=None, names=['data'])

##### Fraction of original network size
x_ax = 75 * np.arange(0, 4)

plt.figure(figsize=[30, 10])
plt.plot(x_ax, large_conn_comp_all.data / large_conn_comp_all.data.max(), linewidth=4)
plt.plot(x_ax, large_conn_comp_regular.data / large_conn_comp_regular.data.max(), linewidth=4)
plt.plot(x_ax, large_conn_comp_elite.data / large_conn_comp_elite.data.max(), linewidth=4)
plt.legend(['Random users', 'Non-elite users', 'Elite users'])
plt.xlabel('Users removed')
plt.ylabel('Fraction of original network size')
plt.savefig('robustness_plot_orig_network.svg', format='svg', bbox_inches="tight")
plt.show()

####### Fraction of original network size lost

x_ax = 75 * np.arange(0, 4)
plt.figure(figsize=[15, 10])
plt.plot(x_ax, 1 - large_conn_comp_all.data / large_conn_comp_all.data.max(), linewidth=4)
plt.plot(x_ax, 1 - large_conn_comp_regular.data / large_conn_comp_regular.data.max(), linewidth=4)
plt.plot(x_ax, 1 - large_conn_comp_elite.data / large_conn_comp_elite.data.max(), linewidth=4)
plt.legend(['Random users', 'Non-elite users', 'Elite users'])
plt.xlabel('Users removed')
plt.ylabel('Fraction of original network size lost')
plt.savefig('robustness_plot_size_lost.svg', format='svg', bbox_inches="tight")
plt.show()

# %%%%%%%%%%%%%%%
import community

best_partition_comm = community.best_partition(social_network)
my_community = list(best_partition_comm.values())



# %%%%%%%%%%%%%%%

def robustness_analysis(my_graph, nodes_, twentyfive_percent_users, verbose=False):
    nodes = copy.copy(nodes_)
    k = 4
    random.shuffle(nodes)

    # Initialize array for LCC sizes
    largest_conn_comp_val = np.zeros(k)

    # Start loop
    for i in range(k):
        # Remove 1 percent of users, randomly chosen
        for j in range(twentyfive_percent_users):
            node = nodes.pop()
            my_graph.remove_node(node)
            
        # Compute largest connected subcomponent
        mcc = len(max(nx.connected_component_subgraphs(my_graph), key=len))
        largest_conn_comp_val[i] = mcc
        
        if verbose:
            print(str(i) + " percent removed.")
            print("Network size: " + str(len(my_graph)))
            print("Largest connected component: " + str(mcc))
    return largest_conn_comp_val

# %%%%%%%%%%%%%%%
# largest connected component

quarter_percent = int(len(nodes_elite_users) * 0.25)

large_conn_comp_elite = robustness_analysis(social_network, nodes_elite_users, quarter_percent)
large_conn_comp_regular = robustness_analysis(social_network, nodes_regular_users, quarter_percent)
large_conn_comp_random = robustness_analysis(social_network, social_network.nodes(), quarter_percent)
largest_conn_comp = pd.DataFrame(np.array([large_conn_comp_elite, large_conn_comp_regular, large_conn_comp_random]).T, columns=['elite', 'reg', 'random'])
largest_conn_comp.to_csv('lcc_only.csv')

# %%%%%%%%%%%%%%%
x_ax = quarter_percent * np.arange(0, 100)
plt.figure(figsize=[15,10])
plt.plot(x_ax, 1 - large_conn_comp_random / large_conn_comp_random.max(), linewidth=4)
plt.plot(x_ax, 1 - large_conn_comp_regular / large_conn_comp_regular.max(), linewidth=4)
plt.plot(x_ax, 1 - large_conn_comp_elite / large_conn_comp_elite.max(), linewidth=4)
plt.legend(['Random users', 'Non-elite users', 'Elite users'])
plt.xlabel('Users removed')
plt.ylabel('Fraction of original network size lost')
plt.savefig('robustness_only_plot.svg', format='svg', bbox_inches="tight")
plt.title('Robustness analysis')
plt.show()

