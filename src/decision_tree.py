import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train_scaled, y_train, depth=None):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train_scaled, y_train)
    return dt

def evaluate_dt(dt, X_test_scaled, y_test, title="Decision tree"):

    y_pred = dt.predict(X_test_scaled)
    y_prob = dt.predict_proba(X_test_scaled)[:, 1]
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

def run_decision_tree(X_train_scaled, y_train, X_test_scaled, y_test):

    depth_list = [3, 5, 7, 9]
    metrics_list = []
   
    print("\n----------DECISION TREE----------")
    for d in depth_list:
         model = train_decision_tree(X_train_scaled, y_train, d)
         print(f"\nDepth : {d}")
         y_pred, y_prob, metrics = evaluate_dt(model, X_test_scaled, y_test)
         metrics_list.append((f"Decision Tree (depth = {d})", metrics))

    return metrics_list


