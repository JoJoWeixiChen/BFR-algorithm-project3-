# BFR-algorithm-project3
The BFR algorithm (short for Bradley-Fayyad-Reina algorithm) is a variant of K-Means algorithm and it performs well on clustering large datasets with limited memory. However, there are three limitations for the BFR algorithm:

1. Each cluster must be normally distributed about a centroid.

2. Different dimensions (features) in each cluster must be independent

3. The algorithm is very sensitive to noise values and outliers.

To implement the BFR algorithm, we developed a model pipeline including 8 steps:

<img width="1067" alt="Screen Shot 2023-04-29 at 9 33 00 PM" src="https://user-images.githubusercontent.com/89158696/235333851-33b4aef5-6a92-4437-a572-8c1034081796.png">

1. **Data Cleaning and pre-processing.** As we mentioned above, the performance of the BFR algorithm can be highly affected by outliers, so we remove the missing values and outliers first. Also, in this step we will split the data into several small chunks and one random sample (size of 5000).

2. **Initialize K (745) clusters.** The BFR algorithm is an iterative algorithm, and the initialized K clusters will also affect the performance of the BFR algorithm. For the initialization, we chose a small sample of data (size of 5000) and performed the K-Means with clusters number as 745 on this sample. And the 745 clusters obtained from the K Means algorithm will be the initialized 745 clusters. The reason why we didn’t pick 745 random points as the initialized 745 clusters is that the strong randomness will lower the robustness of the BFR algorithm.

3. **Load in another chunk of data.**

4. **Assign new points to original clusters and summarize all clusters.** After loading a new chunk of data, we needed to calculate the Mahalanobis distance between each new point (X1, …, Xd) and the centroid of each existing cluster (C1, …, Cd). If the distance is lower than two times of the root of the dimensions (2*32), we could assign the new point into the cluster that satisfies the condition.

5. **Cluster the remaining points, assign new clusters into CS and assign single points into RS.** After step 4, there are still some points from the new chunk that cannot be assigned into any existing clusters. So, we will perform another K-Means algorithm on the remaining data points, assign points that are close enough to each other into new CS and assign the single points into RS.

6. **Merge CS into existing clusters. We will try to merge clusters in CS into the existing clusters based on the combined variance.** In another words, we will try to merge one cluster from CS into an existing cluster, and if the variance of the combined cluster is lower than a number (2*32*variance of the existing cluster), we will combine these two clusters together, otherwise we will regard the two clusters cannot be joined together.

7. **Repeat from step 3 to step 6 until loading the last chunk of data.**

8. **Merge all CS into DS and RS into nearest DS.** After step7, if there are still CS and RS left, we will merge all remaining CS and RS into nearest existing DS based on the combined variance and the Mahalanobis distance mentioned in step 6 and step 4.

In the above steps, here are some definitions for the terminologies:

* DS (Discard set): points closed enough to an existing cluster that can be summarized.

* CS (Compression set): groups of points that are close together but not close to any existing centroids.

* RS (Retained set): Isolated points waiting to be assigned to a compression set.

