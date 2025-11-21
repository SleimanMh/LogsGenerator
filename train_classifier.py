"""
ML Model Training Script - Train classifier to detect error class
"""

import csv
import numpy as np
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


class ErrorClassifier:
    """Train and evaluate error classification model"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.exception_encoder = LabelEncoder()
    
    def load_dataset(self, csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from CSV"""
        X = []
        y = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract features
                features = [
                    self.exception_encoder.fit_transform([row['exception_type']])[0],
                    int(row['depth']),
                    int(row['module_count']),
                    int(row['function_count']),
                    int(row['has_shape_mismatch']),
                    int(row['has_type_error']),
                    int(row['has_value_error']),
                    int(row['has_key_error']),
                    int(row['has_index_error']),
                    int(row['has_device_error']),
                    int(row['has_dimension_error']),
                    int(row['has_encoding_error']),
                ]
                X.append(features)
                y.append(row['error_class'])
        
        self.feature_names = [
            'exception_type', 'depth', 'module_count', 'function_count',
            'has_shape_mismatch', 'has_type_error', 'has_value_error',
            'has_key_error', 'has_index_error', 'has_device_error',
            'has_dimension_error', 'has_encoding_error'
        ]
        
        X = np.array(X)
        y = self.label_encoder.fit_transform(y)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        self.model.fit(X, y)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model on test set"""
        y_pred = self.model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """Perform cross-validation"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'scores': scores,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        }
    
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def predict(self, features: np.ndarray) -> str:
        """Make prediction on new features"""
        pred_idx = self.model.predict(features)[0]
        return self.label_encoder.classes_[pred_idx]
    
    def save(self, filepath: str):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'exception_encoder': self.exception_encoder,
                'feature_names': self.feature_names
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            self.exception_encoder = data['exception_encoder']
            self.feature_names = data['feature_names']
        print(f"Model loaded from {filepath}")


def main():
    """Train error classification model"""
    print("\n" + "="*60)
    print("ERROR CLASSIFICATION MODEL TRAINING")
    print("="*60 + "\n")
    
    # Initialize classifier
    classifier = ErrorClassifier()
    
    # Load dataset
    print("Loading dataset from error_dataset.csv...")
    try:
        X, y = classifier.load_dataset("error_dataset.csv")
        print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")
    except FileNotFoundError:
        print("✗ error_dataset.csv not found. Run error generation endpoints first.")
        return
    
    # Split data
    print("\nSplitting dataset (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining Random Forest model...")
    classifier.train(X_train, y_train)
    print("✓ Model trained")
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = classifier.evaluate(X_test, y_test)
    print(f"✓ Test Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_results = classifier.cross_validate(X, y, cv=5)
    print(f"✓ Mean CV Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    print(f"  Individual scores: {[f'{s:.4f}' for s in cv_results['scores']]}")
    
    # Feature importance
    print("\nFeature Importance:")
    importances = classifier.feature_importance()
    for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        bar_length = int(importance * 50)
        print(f"  {feature:25} {'█' * bar_length} {importance:.4f}")
    
    # Save model
    print("\nSaving model...")
    classifier.save("error_classifier_model.pkl")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60 + "\n")
    
    # Example prediction
    print("Example Prediction:")
    print("-" * 60)
    
    # Create example feature vector
    example_features = np.array([[
        0,  # exception_type (encoded)
        3,  # depth
        2,  # module_count
        1,  # function_count
        0,  # has_shape_mismatch
        1,  # has_type_error
        0,  # has_value_error
        0,  # has_key_error
        0,  # has_index_error
        0,  # has_device_error
        0,  # has_dimension_error
        0,  # has_encoding_error
    ]])
    
    prediction = classifier.predict(example_features)
    print(f"Predicted error class: {prediction}")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
