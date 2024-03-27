from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def apply_model(df, parsed_instructions):

    # Extract the machine learning task, target variable, and features
    ml_task = parsed_instructions.get('TypeOfMachineLearningTask')
    target = parsed_instructions.get('TargetVariable')
    features = parsed_instructions.get('Features')
    print(f"ml_task: {ml_task}")
    print(f"target: {target}")
    print(f"features: {features}")

    # Select only the specified features and the target variable from the DataFrame

    if ml_task == "Predictive Classification":
        df = df[features + target]
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(
            f"Classification accuracy: {accuracy_score(y_test, predictions)}")
        # Plot confusion matrix
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title('Confusion matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    elif ml_task == "Regression Model":
        df = df[features + target]
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"Regression MSE: {mean_squared_error(y_test, predictions)}")
        plt.scatter(y_test, predictions)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()

    elif ml_task == "Cluster Model":
        X = df[target]  # Use only the specified features for clustering

        # Determine the range of possible clusters
        # range_n_clusters = range(2, 10)

        # # Calculate the silhouette score for each number of clusters
        # silhouette_scores = []
        # for n_clusters in range_n_clusters:
        #     model = KMeans(n_clusters=n_clusters)
        #     model.fit(X)
        #     labels = model.labels_
        #     silhouette_scores.append(silhouette_score(X, labels))

        # # Find the number of clusters that has the highest silhouette score
        # optimal_n_clusters = range_n_clusters[silhouette_scores.index(
        #     max(silhouette_scores))]

        # Run KMeans with the optimal number of clusters
        model = KMeans(n_clusters=2)
        model.fit(X)
        labels = model.labels_
        print("Optimal number of clusters: 2")
        print(f"Cluster labels: {labels}")
        # Plot clusters (only works if X has 2 features)
        if len(target) == 2:
            plt.scatter(X[target[0]], X[target[1]], c=labels)
            plt.title('Clusters')
            plt.xlabel(target[0])
            plt.ylabel(target[1])
            plt.show()
    else:
        print(
            "Invalid instructions. Please specify either Predictive Classification, Regression, or Cluster.")
