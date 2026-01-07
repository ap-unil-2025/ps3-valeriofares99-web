from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print(f"{model_name} Results:")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return acc, auc