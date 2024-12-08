import numpy as np
from sklearn.metrics import silhouette_score


def initialize_centroids_kmeans_plus(data, k):
    """
    Initialize centroids using the K-means++ method.
    
    Parameters:
        data: numpy array of data points (n_samples, n_features)
        k: Number of clusters (centroids)
    
    Returns:
        centroids: numpy array of initial centroids
    """
    # Step 1: Randomly select the first centroid
    centroids = [data[np.random.choice(data.shape[0])]]
    
    # Step 2: Select the remaining centroids
    for _ in range(1, k):
        # Calculate the squared distance from each point to the nearest centroid
        distances = np.array([min(np.linalg.norm(point - centroid) ** 2 for centroid in centroids) for point in data])
        
        # Choose the next centroid with a probability proportional to the distance squared
        probabilities = distances / distances.sum()
        next_centroid_idx = np.random.choice(data.shape[0], p=probabilities)
        centroids.append(data[next_centroid_idx])
    
    return np.array(centroids)

# Map step: Assign points in a chunk to the nearest centroid
def map_assign_clusters(chunk, centroids):
    """
    Map step: Assign each point in the chunk to the nearest centroid.
    
    Parameters:
        chunk: numpy array of data points (subset of the dataset)
        centroids: numpy array of current centroids
    
    Returns:
        A list of (cluster_id, point) tuples
    """
    cluster_assignments = []
    for point in chunk:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        cluster_id = np.argmin(distances)
        cluster_assignments.append((cluster_id, point))  # Emit (key, value)
    return cluster_assignments

# Reduce step: Update centroids based on assigned points
def reduce_recalculate_centroids(cluster_assignments, k):
    """
    Reduce step: Recalculate centroids from cluster assignments.
    
    Parameters:
        cluster_assignments: List of (cluster_id, point) tuples
        k: Number of clusters
    
    Returns:
        numpy array of updated centroids
    """
    new_centroids = []
    for i in range(k):
        # Collect all points assigned to cluster i
        cluster_points = np.array([point for cluster_id, point in cluster_assignments if cluster_id == i])
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))  # Average for the new centroid
        else:
            # Handle empty clusters by reinitializing randomly
            new_centroids.append(np.random.rand(cluster_points.shape[1]))
    return np.array(new_centroids)

# Simulated MapReduce process
def mapreduce_kmeans_plus(data, k, max_iters=100, tol=1e-4, chunk_size=100):
    """
    K-Means clustering using a MapReduce-inspired approach with K-means++ initialization.
    
    Parameters:
        data: numpy array of data points
        k: Number of clusters
        max_iters: Maximum number of iterations
        tol: Tolerance for centroid movement to check for convergence
        chunk_size: Number of points per data chunk
    
    Returns:
        Final centroids and cluster labels
    """
    # Step 1: Initialize centroids using K-means++
    centroids = initialize_centroids_kmeans_plus(data, k)
    
    for iteration in range(max_iters):
        print(f"Iteration {iteration + 1}")
        
        # Simulate the Map phase
        cluster_assignments = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            cluster_assignments.extend(map_assign_clusters(chunk, centroids))
        
        # Simulate the Reduce phase
        new_centroids = reduce_recalculate_centroids(cluster_assignments, k)
        
        # Check for convergence (if centroids don't move much)
        if np.all(np.abs(new_centroids - centroids) < tol):
            print("Convergence reached!")
            break
        centroids = new_centroids
    
    # Assign final clusters
    labels = np.zeros(len(data))
    for idx, point in enumerate(data):
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        labels[idx] = np.argmin(distances)
    
    return centroids, labels


# Elbow method for choosing optimal k
def elbow_method_kmeans_plus(data, min_k=1, max_k=10, chunk_size=100):
    """
    Implements the elbow method to choose the optimal number of clusters (k).
    
    Parameters:
        data: numpy array of data points
        min_k: Minimum number of clusters to test
        max_k: Maximum number of clusters to test
        chunk_size: Number of points per data chunk
    
    Returns:
        List of sum of squared distances for each k
    """
    sse = []  # Sum of Squared Errors
    for k in range(min_k, max_k + 1):
        centroids, labels = mapreduce_kmeans_plus(data, k, chunk_size=chunk_size)
        # Compute SSE for this k
        sse_k = np.sum([np.linalg.norm(data[i] - centroids[int(label)])**2 for i, label in enumerate(labels)])
        sse.append(sse_k)
    return sse

# Silhouette method for choosing optimal k
def silhouette_method_kmeans_plus(data, min_k=2, max_k=10, chunk_size=100):
    """
    Implements the silhouette method to evaluate the quality of clustering.
    
    Parameters:
        data: numpy array of data points
        min_k: Minimum number of clusters to test
        max_k: Maximum number of clusters to test
        chunk_size: Number of points per data chunk
    
    Returns:
        List of silhouette scores for each k
    """
    silhouette_scores = []
    for k in range(min_k, max_k + 1):  # Silhouette score requires at least 2 clusters
        centroids, labels = mapreduce_kmeans_plus(data, k, chunk_size=chunk_size)
        # Compute silhouette score for this k
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    return silhouette_scores
