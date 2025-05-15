import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os
import time

# --- Configuration ---
DATA_FILES = {
    "5s": "labled_trafic_5s.csv",
    "1s": "labled_trafic_1s.csv"
}
TARGET_COLUMN = 'label'
DROP_COLUMNS = ['window_start']
INDICATOR_COLUMN = 'is_window_empty'
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_NEIGHBORS = 5
N_ESTIMATORS_RF = 100
PLOTS_DIR = "result_plots"

# --- Model Definitions (same as in training scripts) ---
def get_models():
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    svm = SVC(random_state=RANDOM_STATE, probability=False) # Keep probability False for speed
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE)
    estimators = [
        ('knn', knn),
        ('svm', svm),
        ('rf', rf)
    ]
    voting_clf = VotingClassifier(estimators=estimators, voting='hard')
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    models = {
        "KNN": knn,
        "SVM": svm,
        "Random Forest": rf,
        "Voting Classifier": voting_clf,
        "Stacking Classifier": stacking_clf
    }
    return models

# --- Data Processing and Evaluation Logic ---
def process_and_evaluate(data_label, file_path, models):
    print(f"--- Processing and Evaluating for {data_label} data ({file_path}) ---")
    results = {}
    
    # Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Input file {file_path} not found. Skipping.")
        return None
    print(f"Data loaded. Shape: {df.shape}")

    # Prepare Data
    X = df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS)
    y = df[TARGET_COLUMN]
    X_indicators = X[INDICATOR_COLUMN].values
    if np.any(np.isinf(X)): X = X.replace([np.inf, -np.inf], 99999)
    if X.isnull().values.any(): X.fillna(0, inplace=True)

    # Split Data
    X_train, X_test, y_train, y_test, _, X_test_indicators = train_test_split(
        X, y, X_indicators, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and Evaluate Models
    for name, model in models.items():
        print(f"  Training {name} for {data_label} data...")
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        print(f"  Training finished in {time.time() - start_time:.2f}s")
        y_pred = model.predict(X_test_scaled)
        
        # Store results (metrics report as dict, confusion matrices)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        cm_overall = confusion_matrix(y_test, y_pred)
        
        mask_non_empty = (X_test_indicators == 0)
        cm_non_empty = confusion_matrix(y_test[mask_non_empty], y_pred[mask_non_empty]) if np.any(mask_non_empty) else None
        report_dict_non_empty = classification_report(y_test[mask_non_empty], y_pred[mask_non_empty], output_dict=True, zero_division=0) if np.any(mask_non_empty) else None

        mask_empty = (X_test_indicators == 1)
        cm_empty = confusion_matrix(y_test[mask_empty], y_pred[mask_empty]) if np.any(mask_empty) else None
        report_dict_empty = classification_report(y_test[mask_empty], y_pred[mask_empty], output_dict=True, zero_division=0) if np.any(mask_empty) else None

        results[name] = {
            'report': report_dict,
            'cm_overall': cm_overall,
            'report_non_empty': report_dict_non_empty,
            'cm_non_empty': cm_non_empty,
            'report_empty': report_dict_empty,
            'cm_empty': cm_empty
        }
        print(f"  Evaluation complete for {name}.")
        
    return results

# --- Plotting Function ---
def plot_confusion_matrix(cm, title, filename):
    if cm is None or cm.shape[0] < 2 or cm.shape[1] < 2:
        print(f"Skipping plot {title}: Confusion matrix is None or too small.")
        return
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Human (0)', 'Bot (1)'], 
                yticklabels=['Human (0)', 'Bot (1)'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Saved plot: {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
    plt.close() # Close the plot to free memory

# --- Helper function for formatting metrics ---
def format_metrics_table(report_dict):
    if not report_dict:
        return "N/A"
    
    try:
        accuracy = report_dict.get('accuracy', 'N/A')
        weighted_avg = report_dict.get('weighted avg', {})
        precision = weighted_avg.get('precision', 'N/A')
        recall = weighted_avg.get('recall', 'N/A')
        f1_score = weighted_avg.get('f1-score', 'N/A')

        # Prepare data for table
        metrics_data = [
            ("Accuracy", f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy),
            ("Precision", f"{precision:.4f}" if isinstance(precision, float) else precision),
            ("Recall", f"{recall:.4f}" if isinstance(recall, float) else recall),
            ("F1-Score", f"{f1_score:.4f}" if isinstance(f1_score, float) else f1_score)
        ]
        
        # Basic markdown-like table formatting
        header = "| Metryka   | Wartość |"
        separator = "|-----------|---------|"
        rows = [f"| {name:<9} | {value:<7} |" for name, value in metrics_data]
        
        return "\n".join([header, separator] + rows)
        
    except Exception as e:
        print(f"Error formatting metrics: {e}") # Log error
        return "Error during formatting"

# --- Table Generation Function ---
def generate_detailed_tables(all_results, output_file="classification_reports.txt"):
    print(f"\n--- Generating Detailed Classification Reports (saving to {output_file}) ---")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for time_window, results in all_results.items():
                f.write(f"\n===== Results for {time_window} Window =====\n\n")
                for model_name, data in results.items():
                    f.write(f"--- Model: {model_name} ({time_window} Window) ---\n")

                    # Overall Report
                    f.write("\n  Overall Classification Report:\n")
                    f.write(format_metrics_table(data.get('report')) + "\n")

                    # Non-Empty Report
                    f.write("\n  Non-Empty Windows Classification Report:\n")
                    f.write(format_metrics_table(data.get('report_non_empty')) + "\n")
                        
                    # Empty Report
                    f.write("\n  Empty Windows Classification Report:\n")
                    f.write(format_metrics_table(data.get('report_empty')) + "\n")

                    f.write("\n" + "-" * (len(model_name) + len(time_window) + 20) + "\n") # Separator
                f.write("\n" + "=" * 30 + "\n")
        print(f"Successfully saved detailed reports to {output_file}")
    except Exception as e:
        print(f"Error writing detailed reports to {output_file}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    models = get_models()
    all_results = {}

    # Create directory for plots if it doesn't exist
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"Created directory: {PLOTS_DIR}")

    # Process each dataset
    for data_label, file_path in DATA_FILES.items():
        results = process_and_evaluate(data_label, file_path, models)
        if results:
            all_results[data_label] = results

    # Generate Tables
    if all_results:
        generate_detailed_tables(all_results)

    # Generate Heatmaps
    print("\n--- Generating Confusion Matrix Heatmaps ---")
    if all_results:
        for time_window, results in all_results.items():
            for model_name, data in results.items():
                # Overall CM
                plot_title_overall = f'CM: {model_name} ({time_window} Window) - Overall'
                filename_overall = os.path.join(PLOTS_DIR, f'heatmap_{model_name}_{time_window}_overall.png')
                plot_confusion_matrix(data['cm_overall'], plot_title_overall, filename_overall)
                
                # Non-Empty CM
                plot_title_ne = f'CM: {model_name} ({time_window} Window) - Non-Empty'
                filename_ne = os.path.join(PLOTS_DIR, f'heatmap_{model_name}_{time_window}_non_empty.png')
                plot_confusion_matrix(data['cm_non_empty'], plot_title_ne, filename_ne)
                
                # Empty CM
                plot_title_e = f'CM: {model_name} ({time_window} Window) - Empty'
                filename_e = os.path.join(PLOTS_DIR, f'heatmap_{model_name}_{time_window}_empty.png')
                plot_confusion_matrix(data['cm_empty'], plot_title_e, filename_e)
                
    print("\n--- Visualization Script Finished ---") 