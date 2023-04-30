import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/data/p_dsi/big_data_scaling_sp23/project/Clustering/data.csv')
labels = pd.read_csv('/data/p_dsi/big_data_scaling_sp23/project/Clustering/labels.csv', header = None)

df_sampled = df
labels_sampled = labels
labels_sampled_2 = labels_sampled.copy()
le = LabelEncoder()
labels_sampled_2[1] = le.fit_transform(labels_sampled_2[1])
new_names = {0 : 'contigname'}
labels_sampled_2 = labels_sampled_2.rename(columns=new_names)

def z_score(df):
    df_std = df.copy()
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()    
    return df_std  
df_sampled_num = df_sampled.drop(['contigname'], axis = 1)
df_standardized = z_score(df_sampled_num)

n_cluster = 745
chunks = 50
#np.random.seed(53)
def group_cluster(labels, data):
    assert len(labels) == data.shape[0]
    idx = np.arange(data.shape[0])
    clusters_dict = dict()
    for l, i in zip(labels, idx):
        if l in clusters_dict:
            clusters_dict[l].append(i)
        else:
            clusters_dict[l] = [i]

    return clusters_dict

def process_cluster(cluster_dict):
    #others - Points of clusters with multiple points in one cluster.
    single_cluster_points, others = list(), list()
    for k, v in cluster_dict.items():
        if len(v) == 1:
            single_cluster_points.extend(v)
        else:
            others.extend(v)
    return single_cluster_points, others

def group_process_cluster(labels, idx):
    cluster_dict = group_cluster(labels, idx)
    return process_cluster(cluster_dict)

def summarise_clusters(data, labels):
    assert data.shape[0] == len(labels)
    compressed_clusters = []
    cluster_dict = group_cluster(labels, data)

    for k, v in cluster_dict.items():
        compressed_clusters.append({
            'n' : len(v),
            'sum' : data[v, :].sum(axis=0),
            'sumsq' : np.square(data[v, :]).sum(axis=0)
        })

    return compressed_clusters

def cluster_kmeans(n_cluster, data):
    return KMeans(n_clusters=n_cluster).fit(data)

def mahalanobis(x, c, sigma):
    return np.sqrt(np.sum(np.square((x - c)/sigma)))

def mahalanobis_distance(cluster, point):
    center = cluster['sum'] / cluster['n']
    temp = (cluster['sumsq'] / cluster['n']) - np.square(center)
    temp = temp.astype(float)
    sigma = np.sqrt(temp)
    return mahalanobis(point, center, sigma)

def add_point_cluster(point, cluster):
    cluster['n'] += 1
    cluster['sum'] += point
    cluster['sumsq'] += np.square(point)
    return cluster

def add_to_clusters(points, clusters):
    # Adds points to nearest clusters if less than 2root d and returns unassigned points
    threshold = 2 * np.sqrt(points.shape[1])
    unassigned = list()
    for p in range(len(points)):
        flag = False
        for i in range(len(clusters)):
            d = mahalanobis_distance(clusters[i], points[p])
            if d < threshold:
                clusters[i] = add_point_cluster(points[p], clusters[i])
                flag = True
                break
        if not flag:
            unassigned.append(p)
    return unassigned

def distance_bw_clusters(c1, c2):
    center1 = c1['sum'] / c1['n']
    center2 = c2['sum'] / c2['n']
    sd1 = np.sqrt((c1['sumsq'] / c1['n']) - np.square(center1))
    sd2 = np.sqrt((c2['sumsq'] / c2['n']) - np.square(center2))
    return mahalanobis(center1, center2, sd2*sd1)

def join_clusters(c1, c2):
    c1['n'] += c2['n']
    c1['sum'] += c2['sum']
    c1['sumsq'] += c2['sumsq']
    return c1

def count_points(cluster_dicts):
    return sum([cluster_dict['n'] for cluster_dict in cluster_dicts])

def merge_clusters(old_clusters, new_clusters, threshold, return_two):
    merge_tuples = list()
    for j in range(len(new_clusters)):
        for i in range(len(old_clusters)):
            d = distance_bw_clusters(old_clusters[i], new_clusters[j])
            if d < threshold:
                merge_tuples.append((i, j))
                break

    for i, j in merge_tuples:
        old_clusters[i] = join_clusters(old_clusters[i], new_clusters[j])

    for i, j in sorted(merge_tuples, key=lambda x:x[1], reverse=True):
        new_clusters.pop(j)

    if return_two:
        return old_clusters, new_clusters
    else:
        old_clusters.extend(new_clusters)
        return old_clusters

def print_stats(ds_clusters, cs_clusters, rs_points):
    stats = generate_stats(ds_clusters, cs_clusters, rs_points)
    stats['SUM'] = stats['DS Points'] + stats['CS Points'] + stats['RS Points']
    print(stats)
    return stats

def generate_stats(ds_clusters, cs_clusters, rs_points):
    return {
        'DS Clusters' : len(ds_clusters),
        'DS Points' : count_points(ds_clusters),
        'CS Clusters': len(cs_clusters),
        'CS Points': count_points(cs_clusters),
        'RS Points': len(rs_points),
    }

def predict(clusters, data, threshold):
    final_labels = []
    for chunk in range(chunks):
        for p in range(len(data[chunk])):
            min_d, assignment = None, None
            for c in range(len(clusters)):
                d = mahalanobis_distance(clusters[c], data[chunk][p,2:])
                if min_d is None:
                    min_d, assignment = d, c
                    continue
                if d < min_d:
                    min_d, assignment = d, c

            if min_d < threshold:
                final_labels.append([data[chunk][p,1], assignment])
            else:
                final_labels.append([data[chunk][p,1], -1])
    #final_labels = sorted(final_labels, key=lambda x:x[0])
    #print(final_labels)
    return final_labels


def main(data):
    # Step 1 : Load data into 5 chunks
    sum_ = 0
    r, c = data[0].shape
    threshold = 2*np.sqrt(data[0].shape[1]-2)

    stats = []
    cs_clusters, rs_points, new_cs_clusters = [], None, []

    # Step 2 : Run K-Means with large K(5 times the number of clusters)
    kmeans = cluster_kmeans(5*n_cluster, data[0][:, 2:])

    # Step 3 : Move points with belong to cluster with one point
    rs_idx, other_idx = group_process_cluster(kmeans.labels_, data[0][:, 2:])
    if len(rs_idx):
        rs_points = data[0][rs_idx, 2:]

    # Step 4 & 5: Run Kmeans on rest of data and generate DS Clusters
    kmeans = cluster_kmeans(n_cluster, data[0][other_idx, 2:])
    ds_clusters = summarise_clusters(data[0][other_idx, 2:], kmeans.labels_)

    if len(rs_idx):
        # Step 6: Cluster with large number, since rs_points are very less, we use smaller k
        kmeans = cluster_kmeans(int(len(rs_points)/5), rs_points)
        rs_idx, cs_idx = group_process_cluster(kmeans.labels_, rs_points)

    if len(cs_idx):
        cs_clusters = summarise_clusters(rs_points[cs_idx, :], kmeans.labels_[cs_idx])

    if len(rs_idx):
        rs_points = rs_points[rs_idx, :]

    stats.append(print_stats(ds_clusters, cs_clusters, rs_points))
    # Step 7: load next chunk
    for i in range(1, chunks):

        # Step 8: Assign new points to nearest DS clusters and filter unassigned
        unassigned = add_to_clusters(data[i][:, 2:], ds_clusters)

        # Step 9 : Unassigned points are added to nearest CS clusters
        if len(unassigned):
            unassigned = add_to_clusters(data[i][unassigned, 2:], cs_clusters)

        # Step 10: Unassigned points are added to rs
        if len(unassigned):
            rs_points = np.append(rs_points, data[i][unassigned, 2:], axis=0)


        # Step 11 : Run kmeans on RS points and create CS cluster and RS points
        if len(rs_points) >= 5*n_cluster:
            kmeans = cluster_kmeans(n_cluster, rs_points)
            rs_idx, cs_idx = group_process_cluster(kmeans.labels_, rs_points)
            if len(cs_idx):
                new_cs_clusters = summarise_clusters(rs_points[cs_idx], kmeans.labels_[cs_idx])

            if len(rs_idx):
                rs_points = rs_points[rs_idx]

            # Step 12: Merge CS Clusters with less than threshold distance
            if len(new_cs_clusters):
                cs_clusters = merge_clusters(cs_clusters, new_cs_clusters, threshold, return_two=False)

        if i < chunks-1:
            stats.append(print_stats(ds_clusters, cs_clusters, rs_points))

    # Merge cs and ds clusters
    ds_clusters, cs_clusters = merge_clusters(ds_clusters, cs_clusters, threshold, return_two=True)
    stats.append(print_stats(ds_clusters, cs_clusters, rs_points))

    # Generate predictions
    predictions = predict(ds_clusters, data, threshold)
    return predictions

df_standardized['contigname'] = df_sampled['contigname']
df_standardized = df_standardized.reset_index(drop = False)
cols = df_standardized.columns.tolist()
cols.insert(1, cols.pop(-1))
df_standardized  = df_standardized[cols]
df_standardizednp = df_standardized.to_numpy()
data_model = np.array_split(df_standardizednp, chunks)

result = main(data_model)

data = pd.DataFrame(result, columns = ['contigname', 'predicted_label'])
df_joined = data.merge(labels_sampled_2, how = "inner", on = "contigname")
df_joined = df_joined.rename(columns = {1 : 'actual_label'})
print(adjusted_rand_score(df_joined['predicted_label'], df_joined['actual_label']))