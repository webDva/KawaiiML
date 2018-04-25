import numpy

class KMeansClustering:
    """Class for performing k-means clustering on a set of observations."""

    def __init__(self, observations = None, k = None):
        self.observations = observations
        self.k = k

        # see if the number of observations is greater than or equal to the number of clusters
        if observations.size < k:
            raise ValueError('Number of clusters %d is greater than the number of observations %d.' % (k, observations.size))

        # see if observations is one-dimensional
        if observations.ndim != 1:
            raise ValueError('Dimension of observations is %d, expected 1' % (observations.ndim))

    def hard_kmeans(self):
        """Uses the Lloyod's algorithm to perform hard k-means clustering."""

        iterations = 0

        # initialize with random cluster centroids using the Forgy method
        cluster_means = []
        for i in range(self.k):
            while True: # for preventing duplicates
                random_observation = numpy.random.choice(self.observations)
                if random_observation in cluster_means:
                    continue
                cluster_means.append(random_observation)
                break

        while True:
            # assignment step
            clusters = [[] for i in range(self.k)]
            for x in self.observations:
                closest_cluster_index = numpy.argmin([numpy.linalg.norm(x - mean) ** 2 for mean in cluster_means])
                clusters[closest_cluster_index].append(x)

            # udpate step
            new_cluster_means = [[] for i in range(self.k)]
            for mean in cluster_means:
                new_cluster_means[cluster_means.index(mean)] = numpy.sum(clusters[cluster_means.index(mean)]) / len(clusters[cluster_means.index(mean)])

            # check for convergence
            iterations += 1
            if new_cluster_means == cluster_means:
                break
            cluster_means = new_cluster_means

        return (cluster_means, clusters, iterations)