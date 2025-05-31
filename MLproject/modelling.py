import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Credit Approval Basic Models V1")

# Load data
print("Loading data...")
train_data = pd.read_csv('processed/cleaned_training.csv')
test_data = pd.read_csv('processed/cleaned_testing.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Identify target column for Credit Approval (binary classification)
target_candidates = ['SeriousDlqin2yrs', 'target', 'approval', 'approved', 'credit_approval', 'default']
target_column = None

for col in target_candidates:
    if col in train_data.columns:
        target_column = col
        break

if target_column is None:
    print("Available columns:", list(train_data.columns))
    # Assume first column or last column as target
    target_column = train_data.columns[0]  # Try first column for credit approval
    print(f"Using '{target_column}' as target column")

print(f"Target column: {target_column}")
print(f"Target distribution (Credit Approval):")
target_counts = train_data[target_column].value_counts()
print(target_counts)
print(f"Class balance - 0 (Rejected): {target_counts.get(0, 0)}, 1 (Approved): {target_counts.get(1, 0)}")

# Check for class imbalance
total_samples = len(train_data)
approval_rate = target_counts.get(1, 0) / total_samples
print(f"Credit Approval Rate: {approval_rate:.2%}")

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
        stratify=y_train  # Maintain class balance in split
    )

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Training set approval rate: {y_train.mean():.2%}")
print(f"Test set approval rate: {y_test.mean():.2%}")

# Create input example for MLflow
input_example = X_train.iloc[0:5]

# Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store models and results
models_results = {}

print("\n" + "="*60)
print("TRAINING CREDIT APPROVAL MODELS")
print("="*60)

# 1. Random Forest - Good for credit approval with interpretability
print("\n1. Training Random Forest...")
with mlflow.start_run(run_name="Random_Forest_Credit_Approval"):
    # Optimized parameters for credit approval
    rf_params = {
        'n_estimators': 200,  # More trees for better performance
        'max_depth': 10,      # Limit depth to prevent overfitting
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',  # Handle class imbalance
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Log parameters
    mlflow.log_params(rf_params)
    mlflow.log_param("model_type", "RandomForest_CreditApproval")
    mlflow.log_param("task", "Credit Approval Binary Classification")
    
    # Train model
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    rf_accuracy = rf_model.score(X_test, y_test)
    rf_predictions = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability of approval
    
    # Additional metrics for credit approval
    rf_precision = precision_score(y_test, rf_predictions)
    rf_recall = recall_score(y_test, rf_predictions)
    rf_f1 = f1_score(y_test, rf_predictions)
    rf_auc = roc_auc_score(y_test, rf_proba)
    
    # Log metrics
    mlflow.log_metric("accuracy", rf_accuracy)
    mlflow.log_metric("precision", rf_precision)
    mlflow.log_metric("recall", rf_recall)
    mlflow.log_metric("f1_score", rf_f1)
    mlflow.log_metric("auc_roc", rf_auc)
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=rf_model,
        artifact_path="model",
        input_example=input_example
    )
    
    models_results['Random Forest'] = {
        'model': rf_model,
        'accuracy': rf_accuracy,
        'precision': rf_precision,
        'recall': rf_recall,
        'f1_score': rf_f1,
        'auc_roc': rf_auc,
        'predictions': rf_predictions,
        'probabilities': rf_proba
    }
    
    print(f"Random Forest - Accuracy: {rf_accuracy:.4f}, AUC: {rf_auc:.4f}")

# 2. Logistic Regression - Standard for credit approval
print("\n2. Training Logistic Regression...")
with mlflow.start_run(run_name="Logistic_Regression_Credit_Approval"):
    # Parameters optimized for credit approval
    lr_params = {
        'random_state': 42,
        'max_iter': 2000,
        'solver': 'liblinear',
        'class_weight': 'balanced',  # Handle class imbalance
        'C': 1.0  # Regularization parameter
    }
    
    # Log parameters
    mlflow.log_params(lr_params)
    mlflow.log_param("model_type", "LogisticRegression_CreditApproval")
    mlflow.log_param("task", "Credit Approval Binary Classification")
    
    # Train model
    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    lr_accuracy = lr_model.score(X_test_scaled, y_test)
    lr_predictions = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Additional metrics
    lr_precision = precision_score(y_test, lr_predictions)
    lr_recall = recall_score(y_test, lr_predictions)
    lr_f1 = f1_score(y_test, lr_predictions)
    lr_auc = roc_auc_score(y_test, lr_proba)
    
    # Log metrics
    mlflow.log_metric("accuracy", lr_accuracy)
    mlflow.log_metric("precision", lr_precision)
    mlflow.log_metric("recall", lr_recall)
    mlflow.log_metric("f1_score", lr_f1)
    mlflow.log_metric("auc_roc", lr_auc)
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=lr_model,
        artifact_path="model",
        input_example=pd.DataFrame(X_train_scaled[:5], columns=X_train.columns)
    )
    
    models_results['Logistic Regression'] = {
        'model': lr_model,
        'accuracy': lr_accuracy,
        'precision': lr_precision,
        'recall': lr_recall,
        'f1_score': lr_f1,
        'auc_roc': lr_auc,
        'predictions': lr_predictions,
        'probabilities': lr_proba
    }
    
    print(f"Logistic Regression - Accuracy: {lr_accuracy:.4f}, AUC: {lr_auc:.4f}")

# 3. Gradient Boosting - Excellent for credit approval
print("\n3. Training Gradient Boosting...")
with mlflow.start_run(run_name="GradientBoosting_Credit_Approval"):
    # Parameters for credit approval
    gb_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42
    }
    
    # Log parameters
    mlflow.log_params(gb_params)
    mlflow.log_param("model_type", "GradientBoosting_CreditApproval")
    mlflow.log_param("task", "Credit Approval Binary Classification")
    
    # Train model
    gb_model = GradientBoostingClassifier(**gb_params)
    gb_model.fit(X_train, y_train)
    
    # Evaluate model
    gb_accuracy = gb_model.score(X_test, y_test)
    gb_predictions = gb_model.predict(X_test)
    gb_proba = gb_model.predict_proba(X_test)[:, 1]
    
    # Additional metrics
    gb_precision = precision_score(y_test, gb_predictions)
    gb_recall = recall_score(y_test, gb_predictions)
    gb_f1 = f1_score(y_test, gb_predictions)
    gb_auc = roc_auc_score(y_test, gb_proba)
    
    # Log metrics
    mlflow.log_metric("accuracy", gb_accuracy)
    mlflow.log_metric("precision", gb_precision)
    mlflow.log_metric("recall", gb_recall)
    mlflow.log_metric("f1_score", gb_f1)
    mlflow.log_metric("auc_roc", gb_auc)
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=gb_model,
        artifact_path="model",
        input_example=input_example
    )
    
    models_results['Gradient Boosting'] = {
        'model': gb_model,
        'accuracy': gb_accuracy,
        'precision': gb_precision,
        'recall': gb_recall,
        'f1_score': gb_f1,
        'auc_roc': gb_auc,
        'predictions': gb_predictions,
        'probabilities': gb_proba
    }
    
    print(f"Gradient Boosting - Accuracy: {gb_accuracy:.4f}, AUC: {gb_auc:.4f}")

# 4. XGBoost - State-of-the-art for credit approval
print("\n4. Training XGBoost...")
with mlflow.start_run(run_name="XGBoost_Credit_Approval"):
    # Calculate scale_pos_weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    # Parameters for credit approval
    xgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    # Log parameters
    mlflow.log_params(xgb_params)
    mlflow.log_param("model_type", "XGBoost_CreditApproval")
    mlflow.log_param("task", "Credit Approval Binary Classification")
    
    # Train model
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train)
    
    # Evaluate model
    xgb_accuracy = xgb_model.score(X_test, y_test)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Additional metrics
    xgb_precision = precision_score(y_test, xgb_predictions)
    xgb_recall = recall_score(y_test, xgb_predictions)
    xgb_f1 = f1_score(y_test, xgb_predictions)
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    
    # Log metrics
    mlflow.log_metric("accuracy", xgb_accuracy)
    mlflow.log_metric("precision", xgb_precision)
    mlflow.log_metric("recall", xgb_recall)
    mlflow.log_metric("f1_score", xgb_f1)
    mlflow.log_metric("auc_roc", xgb_auc)
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=xgb_model,
        artifact_path="model",
        input_example=input_example
    )
    
    models_results['XGBoost'] = {
        'model': xgb_model,
        'accuracy': xgb_accuracy,
        'precision': xgb_precision,
        'recall': xgb_recall,
        'f1_score': xgb_f1,
        'auc_roc': xgb_auc,
        'predictions': xgb_predictions,
        'probabilities': xgb_proba
    }
    
    print(f"XGBoost - Accuracy: {xgb_accuracy:.4f}, AUC: {xgb_auc:.4f}")

# Compare all models
print("\n" + "="*60)
print("CREDIT APPROVAL MODELS COMPARISON")
print("="*60)

# Sort models by AUC-ROC (better metric for binary classification)
sorted_models_auc = sorted(models_results.items(), key=lambda x: x[1]['auc_roc'], reverse=True)
sorted_models_f1 = sorted(models_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)

print("Ranking by AUC-ROC Score:")
print("-" * 40)
for i, (name, results) in enumerate(sorted_models_auc, 1):
    print(f"{i}. {name}: AUC={results['auc_roc']:.4f}, Acc={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")

print("\nRanking by F1-Score:")
print("-" * 40)
for i, (name, results) in enumerate(sorted_models_f1, 1):
    print(f"{i}. {name}: F1={results['f1_score']:.4f}, AUC={results['auc_roc']:.4f}, Acc={results['accuracy']:.4f}")

# Best model based on AUC-ROC
best_model_name, best_model_results = sorted_models_auc[0]
print(f"\nBest Model for Credit Approval: {best_model_name}")
print(f"AUC-ROC: {best_model_results['auc_roc']:.4f}")
print(f"Accuracy: {best_model_results['accuracy']:.4f}")
print(f"Precision: {best_model_results['precision']:.4f}")
print(f"Recall: {best_model_results['recall']:.4f}")
print(f"F1-Score: {best_model_results['f1_score']:.4f}")

# Feature importance for tree-based models
if 'Random Forest' in models_results or 'XGBoost' in models_results:
    print(f"\nTop 10 Feature Importance for Credit Approval:")
    print("-" * 50)
    
    if 'XGBoost' in models_results:
        xgb_model = models_results['XGBoost']['model']
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("XGBoost Feature Importance:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    elif 'Random Forest' in models_results:
        rf_model = models_results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Random Forest Feature Importance:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Detailed report for best model
print(f"\nDetailed Classification Report for {best_model_name}:")
print("-" * 60)
print(classification_report(y_test, best_model_results['predictions'], 
                          target_names=['Rejected (0)', 'Approved (1)']))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, best_model_results['predictions'])
print(f"\nConfusion Matrix for {best_model_name}:")
print("-" * 30)
print(f"True Negatives (Correct Rejections): {cm[0,0]}")
print(f"False Positives (Wrong Approvals): {cm[0,1]}")
print(f"False Negatives (Wrong Rejections): {cm[1,0]}")
print(f"True Positives (Correct Approvals): {cm[1,1]}")

# Business Impact Analysis
total_predictions = len(y_test)
approval_predictions = best_model_results['predictions'].sum()
actual_approvals = y_test.sum()

print(f"\nBusiness Impact Analysis:")
print("-" * 30)
print(f"Total Applications: {total_predictions}")
print(f"Model Approved: {approval_predictions} ({approval_predictions/total_predictions:.1%})")
print(f"Actually Approved: {actual_approvals} ({actual_approvals/total_predictions:.1%})")
print(f"False Positive Rate (Risk): {cm[0,1]/(cm[0,0]+cm[0,1]):.2%}")
print(f"False Negative Rate (Lost Opportunity): {cm[1,0]/(cm[1,0]+cm[1,1]):.2%}")

print("\n" + "="*60)
print("CREDIT APPROVAL MODELING COMPLETED!")
print("Check MLflow UI at http://127.0.0.1:5000/ for detailed results")
print("="*60)