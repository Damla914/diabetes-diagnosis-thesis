from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from threshold_optimization import evaluate_threshold


def train_logistic_regression(x_train_scaled, y_train, class_weight=None):

   model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)
   model.fit(x_train_scaled, y_train)
   return model

def evaluate_classifier(model, X_test_scaled, y_test):
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nClassification Report",classification_report(y_test, y_pred))
    print("\nConfussion Matrix:",confusion_matrix(y_test, y_pred))

    return y_pred, y_prob, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

def run_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test):

    metrics_list = []
    thresholds = [0.50, 0.45, 0.40]
 
    print("-----------STANDART LOGISTIC REGRESSION----------")
    model = train_logistic_regression(X_train_scaled, y_train, class_weight=None) 
    _, _, metrics = evaluate_classifier(model, X_test_scaled, y_test)
    metrics_list.append(("Logistic Regression", metrics))

    print("\n----------BALANCED LOGISTIC REGRESSION----------")
    model_bal = train_logistic_regression(X_train_scaled, y_train, class_weight="balanced")  
    _, y_prob_bal, metrics_bal = evaluate_classifier(model_bal, X_test_scaled, y_test)
    metrics_list.append(("Balanced Logistic Regression", metrics_bal))
    
    print("\n----------THRESHOLD ANALİZİ-----------")
    for t in thresholds:
        thr_metrics = evaluate_threshold(y_test, y_prob_bal, t)
        metrics_list.append((f"Balanced Logistic Regression (thr = {t})", thr_metrics))

    return metrics_list, model, model_bal

