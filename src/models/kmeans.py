import matplotlib.pyplot as plt
import os

from src.data.load_data import load_data
from src.data.preprocess import (
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_k(X):
    wcss = []

    for k in range(1, 11):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42
        )

        model.fit(X)
        wcss.append(model.inertia_)

    print("\nWCSS Values:")

    for k, value in enumerate(wcss, start=1):
        print(f"K = {k}, WCSS = {value:.2f}")

    plt.figure(figsize=(8, 6))

    plt.plot(
        range(1, 11),
        wcss,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal K")
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show()

    return wcss


def create_model(k):
    return KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42
    )


def train_model(model, X):
    model.fit(X)

    print("\nK-Means trained successfully!")

    return model


def predict_clusters(model, X):
    labels = model.predict(X)

    return labels


def evaluate_model(model, X, labels):
    silhouette = silhouette_score(
        X,
        labels
    )

    print("\nModel Evaluation")
    print("-" * 25)

    print(f"WCSS: {model.inertia_:.2f}")
    print(f"Iterations: {model.n_iter_}")
    print(f"Silhouette Score: {silhouette:.4f}")

    return silhouette


def display_clusters(X, labels, model):
    plt.figure(figsize=(8, 6))

    plt.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=labels,
        cmap="viridis",
        s=30
    )

    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker="X",
        s=200,
        label="Centroids"
    )

    plt.title("K-Means Clustering")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.legend()

    plt.show()


def main():

    # Load Dataset
    df = load_data()

    print("Original Dataset Shape:")
    print(df.shape)

    # Select Features
    X = df.drop(
        columns=[
            "StudentID",
            "PlacementStatus",
            "Salary Package",
            "IsAnomaly"
        ],
        errors="ignore"
    ).copy()

    print("\nK-Means Dataset Shape:")
    print(X.shape)

    # Identify Features
    numerical_features, categorical_features = (
        identify_features(X)
    )

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)

    # Define Categorical Features
    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]

    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]

    # Handle Missing Values
    X, _, _ = handle_missing_values(
        X,
        X,
        numerical_features
    )

    print("\nMissing Value Handling Completed.")

    # Standardization
    X, _, _ = standardize_data(
        X,
        X,
        numerical_features
    )

    print("Standardization Completed.")

    # One-Hot Encoding
    X, _, _ = one_hot_encode_data(
        X,
        X,
        one_hot_features
    )

    print("One-Hot Encoding Completed.")

    # Ordinal Encoding
    X, _, _ = ordinal_encode_data(
        X,
        X,
        ordinal_features
    )

    print("Ordinal Encoding Completed.")

    # Elbow Method
    find_optimal_k(X)

    # Select K
    k = 4

    # Create Model
    model = create_model(k)

    # Train Model
    model = train_model(
        model,
        X
    )

    # Predict Clusters
    labels = predict_clusters(
        model,
        X
    )

    # Evaluate Model
    evaluate_model(
        model,
        X,
        labels
    )

    # Add Cluster Labels
    df["Cluster"] = labels

    print("\nCluster Distribution:")

    print(
        df["Cluster"]
        .value_counts()
        .sort_index()
    )

    # Display Cluster Centers
    print("\nCluster Centers:")

    print(model.cluster_centers_)

    # Display Clusters
    display_clusters(
        X,
        labels,
        model
    )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    output_path = (
        r"C:\Users\karthik\PycharmProjects\Placementpredict"
        r"\app\static\charts\kmeans_results.csv"
    )

    # Create folder if it does not exist
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    # Save CSV
    df.to_csv(
        output_path,
        index=False
    )

    print("\nK-Means results saved successfully!")

    print("Saved at:")

    print(output_path)


if __name__ == "__main__":
    main()