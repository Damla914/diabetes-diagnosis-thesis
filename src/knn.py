import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

def train_knn(X_train_scaled, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    return knn

def evaluate_knn(knn, X_test_scaled, y_test, title="KNN"):
    
    y_pred = knn.predict(X_test_scaled)
    y_prob = knn.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Auc  : {auc:.4f}")

    metrics = {
         "accuracy": accuracy,
         "precision": precision,
         "recall": recall,
         "f1": f1,
         "auc": auc,
    }

    return y_pred, y_prob, metrics


def run_knn(X_train_scaled, y_train, X_test_scaled, y_test):

    k_list = [3, 5, 7, 9]
    metrics_list = []
    
    print("\n----------KNN-----------")
    for k in k_list:
         model = train_knn(X_train_scaled, y_train, k)
         print(f"\nk : {k}")
         y_pred, y_prob, metrics = evaluate_knn(model, X_test_scaled, y_test)
         metrics_list.append((f"KNN (k = {k})", metrics))
     
    return metrics_list

