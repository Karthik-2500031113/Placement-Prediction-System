from src.data.load_data import load_data
from src.data.preprocess import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

def create_model(k):
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage = "ward"
    )
    return model

def train_model(model, X):
    model.fit(X)
    print("\nAgglomerative Clustering completed \n")
    return model

def evaluate_model(model, X):
    labels = model.lables
    silhouette_samples = score(
        X,
        labels
    )
    print("\nSilhouette score ")
    print(score)
    return labels

def display_dendigram(X, cut_distance):
    linked = linkage(
        X,
        method="ward"
    )
