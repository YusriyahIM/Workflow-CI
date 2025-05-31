import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend (cocok untuk ML pipeline)
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Credit Scoring Hyperparameter Tuning")

# Load data
print("Loading data...")
train_data = pd.read_csv('processed/cleaned_training.csv')
test_data = pd.read_csv('processed/cleaned_testing.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Identify target column
target_candidates = ['SeriousDlqin2yrs', 'target', 'Credit_Score', 'default']
target_column = None

for col in target_candidates:
    if col in train_data.columns:
        target_column = col
        break

if target_column is None:
    print("Available columns:", list(train_data.columns))
    target_column = train_data.columns[-1]
    print(f"Using '{target_column}' as target column")

print(f"Target column: {target_column}")

# Prepare data
X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]

# Check if test data has target column
if target_column in test_data.columns:
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
else:
    # Split training data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Create input example for MLflow
input_example = X_train.iloc[0:5]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

# 1. Random Forest Elastic Search (like your example)
print("\n1. Random Forest - Elastic Search Tuning...")

# Define Elastic Search parameters (following your example pattern)
n_estimators_range = np.linspace(50, 500, 5, dtype=int)  # 5 evenly spaced values
max_depth_range = np.linspace(5, 30, 5, dtype=int)  # 5 evenly spaced values

best_accuracy = 0
best_params = {}
best_model = None

print(f"Testing {len(n_estimators_range)} x {len(max_depth_range)} = {len(n_estimators_range) * len(max_depth_range)} combinations")

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"RF_elastic_search_{n_estimators}_{max_depth}"):
            # Enable autolog for automatic parameter and metric logging
            mlflow.autolog()
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            
            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                best_model = model
                
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example
                )
            
            print(f"n_est: {n_estimators}, max_depth: {max_depth}, accuracy: {accuracy:.4f}")

print(f"\nBest Random Forest:")
print(f"Parameters: {best_params}")
print(f"Accuracy: {best_accuracy:.4f}")

# Store best RF model
rf_best_model = best_model
rf_best_params = best_params
rf_best_accuracy = best_accuracy

# 2. Logistic Regression Grid Search
print("\n2. Logistic Regression - Grid Search Tuning...")

with mlflow.start_run(run_name="LR_GridSearch_Tuning"):
    # Parameter grid
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    }
    
    # Grid search
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42),
        lr_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    lr_grid.fit(X_train_scaled, y_train)
    
    # Best model
    lr_best_model = lr_grid.best_estimator_
    lr_best_accuracy = lr_best_model.score(X_test_scaled, y_test)
    
    # Log results
    mlflow.log_params(lr_grid.best_params_)
    mlflow.log_metric("best_cv_score", lr_grid.best_score_)
    mlflow.log_metric("test_accuracy", lr_best_accuracy)
    
    mlflow.sklearn.log_model(
        sk_model=lr_best_model,
        artifact_path="model",
        input_example=pd.DataFrame(X_train_scaled[:5], columns=X_train.columns)
    )
    
    print(f"Best LR Parameters: {lr_grid.best_params_}")
    print(f"Best CV Score: {lr_grid.best_score_:.4f}")
    print(f"Test Accuracy: {lr_best_accuracy:.4f}")

# 3. Decision Tree Grid Search
print("\n3. Decision Tree - Grid Search Tuning...")

with mlflow.start_run(run_name="DT_GridSearch_Tuning"):
    # Parameter grid
    dt_param_grid = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    
    # Grid search
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        dt_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    dt_grid.fit(X_train, y_train)
    
    # Best model
    dt_best_model = dt_grid.best_estimator_
    dt_best_accuracy = dt_best_model.score(X_test, y_test)
    
    # Log results
    mlflow.log_params(dt_grid.best_params_)
    mlflow.log_metric("best_cv_score", dt_grid.best_score_)
    mlflow.log_metric("test_accuracy", dt_best_accuracy)
    
    mlflow.sklearn.log_model(
        sk_model=dt_best_model,
        artifact_path="model",
        input_example=input_example
    )
    
    print(f"Best DT Parameters: {dt_grid.best_params_}")
    print(f"Best CV Score: {dt_grid.best_score_:.4f}")
    print(f"Test Accuracy: {dt_best_accuracy:.4f}")

# Compare all tuned models
print("\n" + "="*60)
print("TUNED MODELS COMPARISON")
print("="*60)

tuned_results = {
    'Random Forest': rf_best_accuracy,
    'Logistic Regression': lr_best_accuracy,
    'Decision Tree': dt_best_accuracy
}

# Sort by accuracy
sorted_tuned = sorted(tuned_results.items(), key=lambda x: x[1], reverse=True)

print("Final Results:")
for i, (name, accuracy) in enumerate(sorted_tuned, 1):
    print(f"{i}. {name}: {accuracy:.4f}")

# Overall best model
best_tuned_name, best_tuned_accuracy = sorted_tuned[0]
print(f"\nOverall Best Tuned Model: {best_tuned_name}")
print(f"Best Accuracy: {best_tuned_accuracy:.4f}")

# Get best model object and parameters
if best_tuned_name == 'Random Forest':
    final_best_model = rf_best_model
    final_best_params = rf_best_params
    predictions = final_best_model.predict(X_test)
elif best_tuned_name == 'Logistic Regression':
    final_best_model = lr_best_model
    final_best_params = lr_grid.best_params_
    predictions = final_best_model.predict(X_test_scaled)
else:  # Decision Tree
    final_best_model = dt_best_model
    final_best_params = dt_grid.best_params_
    predictions = final_best_model.predict(X_test)

print(f"Best Parameters: {final_best_params}")

# Feature importance for tree-based models
if best_tuned_name in ['Random Forest', 'Decision Tree']:
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': final_best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTop 10 Feature Importance ({best_tuned_name}):")
    print("-" * 50)
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

# Detailed classification report
print(f"\nDetailed Classification Report ({best_tuned_name}):")
print("-" * 60)
print(classification_report(y_test, predictions))

print("\n" + "="*60)
print("HYPERPARAMETER TUNING COMPLETED!")
print("Check MLflow UI at http://127.0.0.1:5000/ for detailed results")
print("="*60)