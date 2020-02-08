import sys
import os
import matplotlib.pyplot as plt
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession, Row
from pyspark.sql import SQLContext
import networkx as nx
import matplotlib
import numpy as np

font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 22}
matplotlib.rc('font', **font)


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
df_user_data = user_data.toPandas()
df_review_data = review_data.toPandas()

user_reg = 'user'
user_elite = 'elite_user'
business = 'biz'

# creating separate list of elite and non elite users
elite_user = df_user_data[df_user_data['elite'] != '']
elite_ids = set(elite_user.user_id)

##############################################
# Begin Constructing the Review Network
##############################################


# Create a NetworkX graph for the review network
review_network = nx.Graph()

# For each review, create a node for the user and business and a link between them
for r in df_review_data.itertuples():
    a = Node(r.user_id, user_elite if r.user_id in elite_ids else user_reg)
    b = Node(r.business_id, business)
    review_network.add_edge(a, b, weight=r.stars)

# Show the number of nodes and edges
print('Nodes:', len(review_network.nodes()))
print('Edges:', len(review_network.edges()))

# Review Network Measures

# Separate nodes based on their type
review_biz_nodes = [n for n in list(review_network.nodes()) if n.Type == business]
review_user_nodes = [n for n in list(review_network.nodes()) if n.Type == user_reg]
review_elite_nodes = [n for n in list(review_network.nodes()) if n.Type == user_elite]

degree_plot(review_network, review_user_nodes, title="All Users", filename='reviews_degree_normal_users')

degree_plot(review_network, review_elite_nodes, title="Elite Users", filename='reviews_degree_elite_users')

degree_plot(review_network, review_biz_nodes, title="Businesses", filename='reviews_degree_all_biz')

##################################
# Centrality Measures
##################################


# Clustering Coefficient
L = max(nx.connected_component_subgraphs(review_network), key=len)
print("Nodes in largest subcomponent:", len(L.nodes()))
print("Edges in largest subcomponent:", len(L.edges()))

cluster_coefficient_avg = nx.average_clustering(review_network)
print('Average clustering coefficient . :', cluster_coefficient_avg)

# Eigenvalue centrality
eigen_value = nx.eigenvector_centrality_numpy(review_network)
eigen_elite_user = [eigen_value[n] for n in eigen_value if n.Type == user_elite]
eigen_regular = [eigen_value[n] for n in eigen_value if n.Type == user_reg]
eigen_elite_avg = np.mean(eigen_elite_user)
eigen_regular_avg = np.mean(eigen_regular)
all_user_ev = np.mean(eigen_elite_user + eigen_regular)

# Degree centrality
deg = nx.degree(review_network)
# print(deg)
deg_elite_user = [deg[n] for (n, d) in deg if n.Type == user_elite]
# print(deg_elite_user)
deg_user = [deg[n] for (n, d) in deg if n.Type == user_reg]
elite_avg_deg = np.mean(deg_elite_user)
user_avg_deg = np.mean(deg_user)
all_user_deg = np.mean(deg_elite_user + deg_user)

# Show results
print('Normal user mean EV centrality', eigen_regular_avg)
print('Elite user mean EV centrality', eigen_elite_avg)
print('All users mean EV centrality', all_user_ev)
ratio = eigen_elite_avg / eigen_regular_avg
print('Ratio EV (Elite : Normal): %.2f' % ratio)

# Show results
print('Normal user mean degree centrality', user_avg_deg)
print('Elite user mean degree centrality', elite_avg_deg)
print('All users mean degree centrality', all_user_deg)
ratio = elite_avg_deg / user_avg_deg
print('Ratio degree (Elite : Normal): %.2f' % ratio)

# Create new column in the user data frame
df_user_data['ev'] = 0
eigen_regular = {n.Data: eigen_value[n] for n in eigen_value if (n.Type == user_elite) or (n.Type == user_reg)}

# Insert the eigenvalue of the user in the data frame. This takes several minutes...
i = 1
p = int(len(eigen_regular) / 100)
for idx in eigen_regular:
    if i % p == 0: print('%i percent done' % (i / p))
    eigenvalue = eigen_regular[idx]
    df_user_data.loc[df_user_data.user_id == idx, 'ev'] = eigenvalue
    i += 1

# Plot the eigenvalue of the user vs. the average rating the user
plt.figure(figsize=[15, 10])
plt.scatter(df_user_data.average_stars, df_user_data.ev, edgecolors='black')
plt.xlabel('Yelp average rating')
plt.ylabel('Eigenvalue centrality')
plt.title('Eigenvalue centrality vs. average user rating for Yelp users in Toronto')
plt.savefig('user_rating_ev.svg', format='svg', bbox_inches="tight")
# plt.show()

##########################
# Restaurants and Eigenvalue Centrality
print(eigen_value)

eigen_business = {n.Data: eigen_value[n] for n in eigen_value if n.Type == business}
deg_business = {n.Data: deg[n] for (n, d) in deg if n.Type == business}
df_business_data['ev'] = 0.0

for idx in eigen_business:
    eigenvalue = eigen_business[idx]
    df_business_data.loc[df_business_data.business_id == idx, 'ev'] = eigenvalue

plt.figure(figsize=[15, 10])
plt.scatter(df_business_data.stars, df_business_data.ev, edgecolors='black')
plt.xlabel('Yelp rating')
plt.ylabel('Eigenvalue centrality')
plt.title('Eigenvalue centrality vs Yelp rating for Restaurants')
plt.savefig('business_rating_eigenval.svg', format='svg', bbox_inches="tight")
# plt.show()

plt.figure(figsize=[15, 10])
plt.scatter(df_business_data.ev, df_business_data.review_count, edgecolors='black')
plt.xlabel('Eigen vector centrality score')
plt.ylabel('Review count')
plt.title('Eigen vector centrality vs. number of review for Restaurants')
plt.savefig('biz_ev_count.svg', format='svg', bbox_inches="tight")
# plt.show()

plt.close('all')

############################################################
# Concrete Differences in Ratings of Elite vs Regular users
############################################################

# Get ids of businesses
business_ids = [b.Data for b in review_biz_nodes]

# Only regular user reviews
business_graph_regular = nx.subgraph(review_network, review_user_nodes + review_biz_nodes)
dict_regular_weights = business_graph_regular.degree(review_biz_nodes, weight='weight')
dict_regular_deg = business_graph_regular.degree(review_biz_nodes)
dict_reg_business_ratings = {
    node.Data: dict_regular_weights[node] / dict_regular_deg[node]
    for node in review_biz_nodes
    if dict_regular_deg[node] > 0
}

# Only elite reviews
elite_business_graph = nx.subgraph(review_network, review_elite_nodes + review_biz_nodes)
dict_elite_weights = elite_business_graph.degree(review_biz_nodes, weight='weight')
dict_elite_deg = elite_business_graph.degree(review_biz_nodes)
elite_biz_ratings_dict = {
    node.Data: dict_elite_weights[node] / dict_elite_deg[node]
    for node in review_biz_nodes
    if dict_elite_deg[node] > 0
}

# All user reviews
dict_all_weights = review_network.degree(review_biz_nodes, weight='weight')
dict_all_degrees = review_network.degree(review_biz_nodes)
all_biz_ratings_dict = {
    node.Data: dict_all_weights[node] / dict_all_degrees[node]
    for node in review_biz_nodes
    if dict_all_degrees[node] > 0
}

# Comparison REGULAR AND ELITE
dlt_elite_regular = {
    business_id: elite_biz_ratings_dict[business_id] - dict_reg_business_ratings[business_id]
    for business_id in business_ids
    if business_id in dict_reg_business_ratings.keys() and business_id in elite_biz_ratings_dict.keys()
}

# Comparison ALL AND ELITE
dlt_elite_all = {
    business_id: elite_biz_ratings_dict[business_id] - all_biz_ratings_dict[business_id]
    for business_id in business_ids
    if business_id in all_biz_ratings_dict.keys() and business_id in elite_biz_ratings_dict.keys()
}

# Comparison ALL AND REG
dlt_regular_all = {
    business_id: dict_reg_business_ratings[business_id] - all_biz_ratings_dict[business_id]
    for business_id in business_ids
    if business_id in all_biz_ratings_dict.keys() and business_id in dict_reg_business_ratings.keys()
}

# Elite User Ratings vs All user Ratings
plt.figure(figsize=[20, 5])
plt.hist(np.array(list(dlt_elite_all.values())), bins=200, edgecolor='black')
plt.xlabel('Delta (stars)')
plt.title('Elite reviews compared to all user reviews')
plt.savefig('delta_elite_all.svg', format='svg', bbox_inches="tight")
# plt.show()

# Elite User Ratings vs Only Reg User Ratings
plt.figure(figsize=[20, 5])
plt.hist(np.array(list(dlt_elite_regular.values())), bins=200, edgecolor='black')
plt.xlabel('Delta (stars)')
plt.title('Elite reviews compared to regular user reviews')
plt.savefig('delta_elite_reg.svg', format='svg', bbox_inches="tight")
# plt.show()

# Regular User Ratings vs All User Ratings
plt.figure(figsize=[20, 5])
plt.hist(np.array(list(dlt_regular_all.values())), bins=200, edgecolor='black')
plt.xlabel('Delta (stars)')
plt.title('Regular user reviews compared to all user reviews')
plt.savefig('delta_all_reg.svg', format='svg', bbox_inches="tight")
# plt.show()
