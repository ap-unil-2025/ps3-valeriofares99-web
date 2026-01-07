from src.data_loader import load_and_preprocess
from src.models import (
    train_logistic_regression,
    train_knn,
    train_naive_bayes,
    train_svm,
    train_random_forest,
    train_xgboost,
)
from src.evaluation import evaluate_model

from sklearn.decomposition import PCA


def main():
    print("=" * 60)
    print("Startup Success Prediction")
    print("=" * 60)

    # Load & preprocess data
    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    ) = load_and_preprocess()

    results = {}

    # Models WITHOUT PCA
    print("\n=== MODELS WITHOUT PCA ===")

    lr = train_logistic_regression(X_train_scaled, y_train)
    results["Logistic Regression"] = evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")[0]

    knn = train_knn(X_train_scaled, y_train)
    results["KNN"] = evaluate_model(knn, X_test_scaled, y_test, "KNN")[0]

    nb = train_naive_bayes(X_train_scaled, y_train)
    results["Naive Bayes"] = evaluate_model(nb, X_test_scaled, y_test, "Naive Bayes")[0]

    svm = train_svm(X_train_scaled, y_train)
    results["SVM"] = evaluate_model(svm, X_test_scaled, y_test, "SVM")[0]

    rf = train_random_forest(X_train, y_train)
    results["Random Forest"] = evaluate_model(rf, X_test, y_test, "Random Forest")[0]

    xgb = train_xgboost(X_train, y_train)
    results["XGBoost"] = evaluate_model(xgb, X_test, y_test, "XGBoost")[0]

    # PCA (95% variance)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"\nPCA components retained: {pca.n_components_}")

    # Models WITH PCA
    print("\n=== MODELS WITH PCA ===")

    lr_pca = train_logistic_regression(X_train_pca, y_train)
    results["Logistic Regression (PCA)"] = evaluate_model(lr_pca, X_test_pca, y_test, "Logistic Regression (PCA)")[0]

    knn_pca = train_knn(X_train_pca, y_train)
    results["KNN (PCA)"] = evaluate_model(knn_pca, X_test_pca, y_test, "KNN (PCA)")[0]

    nb_pca = train_naive_bayes(X_train_pca, y_train)
    results["Naive Bayes (PCA)"] = evaluate_model(nb_pca, X_test_pca, y_test, "Naive Bayes (PCA)")[0]

    svm_pca = train_svm(X_train_pca, y_train)
    results["SVM (PCA)"] = evaluate_model(svm_pca, X_test_pca, y_test, "SVM (PCA)")[0]

    rf_pca = train_random_forest(X_train_pca, y_train)
    results["Random Forest (PCA)"] = evaluate_model(rf_pca, X_test_pca, y_test, "Random Forest (PCA)")[0]

    xgb_pca = train_xgboost(X_train_pca, y_train)
    results["XGBoost (PCA)"] = evaluate_model(xgb_pca, X_test_pca, y_test, "XGBoost (PCA)")[0]

    # Find the winner
    winner_model = max(results, key=results.get)
    winner_accuracy = results[winner_model]

    print("\n" + "=" * 60)
    print(f"Winner: {winner_model} ({winner_accuracy:.3f} accuracy)")
    print("=" * 60)


if __name__ == "__main__":
    main()