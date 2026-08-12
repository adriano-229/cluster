from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class HierarchicalClustering:
    def __init__(self, method="single", n_clusters=3):
        self.method = method
        self.n_clusters = n_clusters

        self.scaler = StandardScaler()
        self.linkage_matrix = None
        self.labels_ = None
        self.X = None
        self.cluster_centers_ = None

    def fit(self, X):
        self.X = self.scaler.fit_transform(X)

        # Hierarchical clustering
        self.linkage_matrix = linkage(self.X, method=self.method)

        # Cut dendrogram into n clusters
        self.labels_ = fcluster(
            self.linkage_matrix,
            self.n_clusters,
            criterion="maxclust",
        )

        # Calculate a representative center for each cluster.
        self.cluster_centers_ = []

        for cluster in range(1, self.n_clusters + 1):
            points = self.X[self.labels_ == cluster]
            self.cluster_centers_.append(points.mean(axis=0))

        self.cluster_centers_ = pd.DataFrame(self.cluster_centers_)

        return self

    def predict(self, X):
        X = self.scaler.transform(X)

        predictions = []

        for point in X:
            distances = (
                ((self.cluster_centers_.values - point) ** 2)
                .sum(axis=1)
            )

            cluster = distances.argmin() + 1
            predictions.append(cluster)

        return predictions

    def plot_dendrogram(self):
        plt.figure(figsize=(12, 6))

        dendrogram(self.linkage_matrix, color_threshold=0)

        plt.title(f"{self.method.capitalize()} Linkage")
        plt.xlabel("Samples")
        plt.ylabel("Distance")
        plt.show()


def load_dataset():
    path = Path(__file__).parent / "candy-data.csv"

    df = pd.read_csv(path)

    # Columns that identify/describe the candy but should not
    # participate in clustering.
    excluded = {
        "competitorname",
        "winpercent",
    }

    features = [
        column
        for column in df.columns
        if column not in excluded
    ]

    X = df[features]

    return X


def plot_pca(X, labels, title, X_new=None, new_labels=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    # Original samples
    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
    )

    # Prediction samples
    if X_new is not None:
        X_new_2d = pca.transform(
            scaler.transform(X_new)
        )

        plt.scatter(
            X_new_2d[:, 0],
            X_new_2d[:, 1],
            c=new_labels,
            marker="x",
            s=100,
            linewidths=3,
        )

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


def main():
    X = load_dataset()

    models = {}

    for method, n_clusters in [("single", 1), ("complete", 4), ("average", 4)]:
        model = HierarchicalClustering(
            method=method,
            n_clusters=n_clusters,
        )

        model.fit(X)

        models[method] = model

        print(f"{method.capitalize()} linkage")
        print("Cluster sizes:")
        print(
            pd.Series(model.labels_)
            .value_counts()
            .sort_index()
        )
        print()

        model.plot_dendrogram()
        plot_pca(X, model.labels_, f"{method.capitalize()} Linkage - PCA")


    # new_df = pd.read_csv("validation.csv")
    new_candy = X.iloc[[0]]

    print("Prediction:")

    for method, model in models.items():
        cluster = model.predict(new_candy)[0]

        print(f"{method.capitalize()}: cluster {cluster}")

        plot_pca(
            X,
            model.labels_,
            f"{method.capitalize()} Linkage - PCA",
            X_new=new_candy,
            new_labels=[cluster],
        )


if __name__ == "__main__":
    main()
