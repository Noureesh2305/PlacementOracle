from src.data_loader import load_data
from src.data_exploration import explore_data
from src.preprocessing import preprocess_data
from src.model_training import (
    split_and_scale,
    train_logistic_regression,
    train_decision_tree,
    train_random_forest
)
from src.evaluation import evaluate_model
from src.feature_importance import show_feature_importance
from src.prediction import predict_placement


def main():
    # ---------------------------
    # 1. Load & Explore Data
    # ---------------------------
    df = load_data("data/placement.csv")
    explore_data(df)

    # ---------------------------
    # 2. Preprocess Data
    # ---------------------------
    df = preprocess_data(df)
    print("\nProcessed Data Sample:")
    print(df.head())

    # ---------------------------
    # 3. Train-Test Split & Scaling
    # ---------------------------
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    print("\nTraining samples:", X_train.shape)
    print("Testing samples :", X_test.shape)

    # ---------------------------
    # 4. Logistic Regression
    # ---------------------------
    print("\n--- Logistic Regression ---")
    log_model = train_logistic_regression(X_train, y_train)
    evaluate_model(log_model, X_test, y_test)

    # ---------------------------
    # 5. Decision Tree
    # ---------------------------
    print("\n--- Decision Tree ---")
    dt_model = train_decision_tree(X_train, y_train)
    evaluate_model(dt_model, X_test, y_test)

    # ---------------------------
    # 6. Random Forest
    # ---------------------------
    print("\n--- Random Forest ---")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)

    # ---------------------------
    # 7. Feature Importance
    # ---------------------------
    feature_names = df.drop('status', axis=1).columns
    show_feature_importance(rf_model, feature_names)

    # ---------------------------
    # 8. Test Prediction (NEW STUDENT)
    # ---------------------------
    new_student = {
        'gender': 1,              # Male
        'ssc_p': 82,
        'ssc_b': 1,
        'hsc_p': 78,
        'hsc_b': 1,
        'hsc_s': 'Science',
        'degree_p': 75,
        'degree_t': 'Sci&Tech',
        'workex': 1,
        'etest_p': 70,
        'specialisation': 1,
        'mba_p': 72
    }

    pred, prob = predict_placement(
        rf_model,
        scaler,
        new_student,
        feature_names
    )

    print("\n--- Placement Prediction for New Student ---")
    print("Prediction:", "Placed" if pred == 1 else "Not Placed")
    print("Placement Probability:", round(prob * 100, 2), "%")


# Entry point
if __name__ == "__main__":
    main()
