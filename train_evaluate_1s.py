import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression # For Stacking meta-learner
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.inspection import permutation_importance
import time

# --- Configuration ---
INPUT_CSV = "labled_trafic_1s.csv" # Changed input CSV
TARGET_COLUMN = 'label'
# Columns to drop before training (non-feature columns)
DROP_COLUMNS = ['window_start']
INDICATOR_COLUMN = 'is_window_empty' # To split evaluation
TEST_SIZE = 0.3 # 30% for testing
RANDOM_STATE = 42 # For reproducibility
N_NEIGHBORS = 5 # For KNN
N_ESTIMATORS_RF = 100 # For Random Forest

# --- Helper Functions ---
def evaluate_model(y_true, y_pred, X_test_indicators, model_name):
    """Evaluates the model on the entire test set and subgroups."""
    print(f"--- Evaluation Results for: {model_name} ---")
    
    # Overall Evaluation
    print("\nOverall Performance:")
    # Add zero_division=0 to classification_report to handle cases with no support for a class in subgroups
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) # Use weighted F1 for overall score
    print(f"Overall Weighted F1-Score: {overall_f1:.4f}")

    # Evaluation on Non-Empty Windows
    mask_non_empty = (X_test_indicators == 0)
    if np.any(mask_non_empty):
        print("\nPerformance on NON-EMPTY Windows (is_window_empty=0):")
        y_true_ne = y_true[mask_non_empty]
        y_pred_ne = y_pred[mask_non_empty]
        if len(np.unique(y_true_ne)) > 1: # Check if more than one class present
            print(classification_report(y_true_ne, y_pred_ne, zero_division=0))
            print("Confusion Matrix (Non-Empty):")
            print(confusion_matrix(y_true_ne, y_pred_ne))
        elif len(y_true_ne) > 0: # Only one class present
             print(f"Only one class ({np.unique(y_true_ne)[0]}) present in non-empty test windows.")
             print(f"Accuracy: {accuracy_score(y_true_ne, y_pred_ne):.4f}")
        else:
            print("No non-empty windows in the test set.")
    else:
         print("\nNo non-empty windows found in the test set.")

    # Evaluation on Empty Windows
    mask_empty = (X_test_indicators == 1)
    if np.any(mask_empty):
        print("\nPerformance on EMPTY Windows (is_window_empty=1):")
        y_true_e = y_true[mask_empty]
        y_pred_e = y_pred[mask_empty]
        if len(np.unique(y_true_e)) > 1:
            print(classification_report(y_true_e, y_pred_e, zero_division=0))
            print("Confusion Matrix (Empty):")
            print(confusion_matrix(y_true_e, y_pred_e))
        elif len(y_true_e) > 0:
             print(f"Only one class ({np.unique(y_true_e)[0]}) present in empty test windows.")
             print(f"Accuracy: {accuracy_score(y_true_e, y_pred_e):.4f}")
        else:
             print("No empty windows in the test set.")
    else:
        print("\nNo empty windows found in the test set.")
        
    print("--------------------------------------------------")
    return overall_f1

# --- Main Script ---
if __name__ == "__main__":
    # 1. Load Data
    print(f"Loading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_CSV} not found. Exiting.")
        exit()
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head())

    # 2. Prepare Data
    print("Preparing data...")
    X = df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS)
    y = df[TARGET_COLUMN]
    feature_names = X.columns.tolist() # Store feature names for later
    X_indicators = X[INDICATOR_COLUMN].values # Get indicator column for evaluation splitting
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Check for infinite values introduced during feature extraction (e.g., ratio=9999)
    if np.any(np.isinf(X)):
        print("Warning: Infinite values found in features. Replacing with a large number (e.g., 99999).")
        X = X.replace([np.inf, -np.inf], 99999)
        
    # Check for NaN values before scaling
    if X.isnull().values.any():
        print("Warning: NaN values found in features. Filling with 0.")
        X.fillna(0, inplace=True)

    # 3. Split Data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test, X_train_indicators, X_test_indicators = train_test_split(
        X, y, X_indicators, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # 4. Scale Features
    print("Scaling features...")
    scaler = StandardScaler()
    # Fit only on training data
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform both training and testing data
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames (optional, but can be useful)
    # X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    # X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    # 5. Define Models
    print("Defining models...")
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    # Use probability=True for SVM if soft voting is desired, but it's slower.
    # Stick to default hard voting for now unless results are poor.
    svm = SVC(random_state=RANDOM_STATE) 
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE)
    
    # Define base estimators for ensemble methods
    estimators = [
        ('knn', KNeighborsClassifier(n_neighbors=N_NEIGHBORS)),
        ('svm', SVC(random_state=RANDOM_STATE)), # Consider probability=True if needed
        ('rf', RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE))
    ]

    # Voting Classifier (Hard Voting by default)
    voting_clf = VotingClassifier(estimators=estimators)
    
    # Stacking Classifier (using Logistic Regression as the final meta-learner)
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    models = {
        "KNN": knn,
        "SVM": svm,
        "Random Forest": rf,
        "Voting Classifier": voting_clf,
        "Stacking Classifier": stacking_clf
    }

    results = {}

    # 6. Train and Evaluate Models
    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds.")

        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_scaled)
        
        # Pass the indicator column from the original *unscaled* test features
        f1 = evaluate_model(y_test, y_pred, X_test_indicators, name)
        results[name] = f1

    # 7. Identify Best Model
    print("\n--- Model Comparison ---")
    best_model_name = max(results, key=results.get)
    print(f"Best model based on overall weighted F1-score: {best_model_name} (F1 = {results[best_model_name]:.4f})")

    # 8. Feature Importance for Best Model
    print(f"\n--- Feature Importance Analysis for {best_model_name} ---")
    best_model = models[best_model_name]

    if hasattr(best_model, 'feature_importances_'): # Typically for tree-based models like RF
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        print("Feature Importances (from model attribute):")
        print(feature_importance_df.head(10)) # Print top 10
    elif hasattr(best_model, 'estimators_') and isinstance(best_model, (VotingClassifier, StackingClassifier)):
        # For ensembles, importance is more complex. We can try permutation importance
        # or look at the importances of the base models if they provide it (e.g., RF in the ensemble)
        print("Feature importance for ensemble models like Voting/Stacking is complex.")
        print("Attempting Permutation Importance...")
        # Make sure to use the scaled test data for permutation importance
        perm_importance = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        print("Feature Importances (Permutation Importance on Test Set):")
        for i in sorted_idx[:10]: # Print top 10
             print(f"{feature_names[i]:<30}: {perm_importance.importances_mean[i]:.4f} +/- {perm_importance.importances_std[i]:.4f}")
    else: # Use permutation importance for models like KNN, SVM
        print("Using Permutation Importance...")
        # Make sure to use the scaled test data for permutation importance
        perm_importance = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        print("Feature Importances (Permutation Importance on Test Set):")
        for i in sorted_idx[:10]: # Print top 10
             print(f"{feature_names[i]:<30}: {perm_importance.importances_mean[i]:.4f} +/- {perm_importance.importances_std[i]:.4f}")

    print("\n--- Script Finished ---") 