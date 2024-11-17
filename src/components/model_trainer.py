import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import joblib
from src.components.data_preprocessing import TextPreprocessor
from src.utils.logger import logging

class ModelTrainer:
    def __init__(self):
        self.models = {
            'recommendation': {
                'rf': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=10,
                    random_state=42
                ),
                'gb': GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ),
                'lr': LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ),
                'nb': MultinomialNB()
            },
            'sentiment': {
                'rf': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=10,
                    random_state=42
                ),
                'gb': GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ),
                'lr': LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            }
        }
        
        self.sampling_strategies = {
            'none': None,
            'smote': SMOTE(random_state=42),
            'undersample': RandomUnderSampler(random_state=42),
            'combined': Pipeline([
                ('smote', SMOTE(sampling_strategy=0.6, random_state=42)),
                ('undersample', RandomUnderSampler(sampling_strategy=0.7, random_state=42))
            ])
        }

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, task='recommendation'):
        """Evaluate a single model"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {}
        if task == 'recommendation':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        else:  # sentiment
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        
        # Add cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        metrics['cv_score_mean'] = cv_scores.mean()
        metrics['cv_score_std'] = cv_scores.std()
        
        return metrics, model

    def train_models(self, save_dir="models"):
        try:
            print("Starting model training...")
            logging.info("Starting model training...")
            
            # Create models directory
            os.makedirs(save_dir, exist_ok=True)
            
            # Load and preprocess data
            print("Loading and preprocessing data...")
            df = pd.read_csv('./data/Womens Clothing E-Commerce Reviews.csv')
            df = df.drop("Unnamed: 0", axis=1) if "Unnamed: 0" in df.columns else df
            df = df.dropna(subset=['Review Text', 'Recommended IND', 'Rating']).reset_index(drop=True)
            
            preprocessor = TextPreprocessor()
            df['cleaned_text'] = df['Review Text'].apply(preprocessor.clean_text)
            df = df[df['cleaned_text'] != ""].reset_index(drop=True)
            
            # Prepare data
            X = df['cleaned_text']
            y_recommend = df['Recommended IND']
            y_sentiment = df['Rating']
            
            # Split data
            X_train, X_test, y_rec_train, y_rec_test, y_sent_train, y_sent_test = train_test_split(
                X, y_recommend, y_sentiment, test_size=0.2, random_state=42
            )
            
            # Vectorize text
            tfidf = TfidfVectorizer(max_features=5000)
            X_train_tfidf = tfidf.fit_transform(X_train)
            X_test_tfidf = tfidf.transform(X_test)
            
            # Train and evaluate models
            results = {
                'recommendation': {},
                'sentiment': {}
            }
            
            # Train recommendation models
            print("\nTraining recommendation models...")
            for model_name, model in self.models['recommendation'].items():
                print(f"\nTraining {model_name}...")
                
                for sampling_name, sampler in self.sampling_strategies.items():
                    print(f"Using {sampling_name} sampling...")
                    
                    # Apply sampling if specified
                    if sampler:
                        X_train_sampled, y_rec_train_sampled = sampler.fit_resample(
                            X_train_tfidf, y_rec_train
                        )
                    else:
                        X_train_sampled, y_rec_train_sampled = X_train_tfidf, y_rec_train
                    
                    metrics, trained_model = self.evaluate_model(
                        model, X_train_sampled, X_test_tfidf,
                        y_rec_train_sampled, y_rec_test,
                        'recommendation'
                    )
                    
                    results['recommendation'][f"{model_name}_{sampling_name}"] = {
                        'metrics': metrics,
                        'model': trained_model
                    }
            
            # Train sentiment models
            print("\nTraining sentiment models...")
            for model_name, model in self.models['sentiment'].items():
                print(f"\nTraining {model_name}...")
                metrics, trained_model = self.evaluate_model(
                    model, X_train_tfidf, X_test_tfidf,
                    y_sent_train, y_sent_test,
                    'sentiment'
                )
                
                results['sentiment'][model_name] = {
                    'metrics': metrics,
                    'model': trained_model
                }
            
            # Select best models
            best_rec_model = self._select_best_model(results['recommendation'], 'recommendation')
            best_sent_model = self._select_best_model(results['sentiment'], 'sentiment')
            
            # Save models
            print("\nSaving models...")
            joblib.dump(best_rec_model, os.path.join(save_dir, "recommendation_model.pkl"))
            joblib.dump(best_sent_model, os.path.join(save_dir, "sentiment_model.pkl"))
            joblib.dump(tfidf, os.path.join(save_dir, "tfidf_vectorizer.pkl"))
            
            print("Training complete!")
            return results, tfidf
            
        except Exception as e:
            error_msg = f"Error in training: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            raise e

    def _select_best_model(self, results, task):
        """Select the best model based on metrics"""
        if task == 'recommendation':
            # For recommendation, use F1 score and CV score
            scores = {
                name: (res['metrics']['f1'] + res['metrics']['cv_score_mean']) / 2
                for name, res in results.items()
            }
        else:
            # For sentiment, use negative RMSE (higher is better) and CV score
            scores = {
                name: (res['metrics']['cv_score_mean'] - res['metrics']['rmse'])
                for name, res in results.items()
            }
        
        best_model_name = max(scores, key=scores.get)
        return results[best_model_name]['model']

def print_results(results):
    """Print training results in a formatted way"""
    print("\nRecommendation Models:")
    print("-" * 50)
    for model_name, data in results['recommendation'].items():
        metrics = data['metrics']
        print(f"\n{model_name}:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1']:.3f}")
        print(f"CV Score: {metrics['cv_score_mean']:.3f} (±{metrics['cv_score_std']:.3f})")
    
    print("\nSentiment Models:")
    print("-" * 50)
    for model_name, data in results['sentiment'].items():
        metrics = data['metrics']
        print(f"\n{model_name}:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"MAE: {metrics['mae']:.3f}")
        print(f"RMSE: {metrics['rmse']:.3f}")
        print(f"CV Score: {metrics['cv_score_mean']:.3f} (±{metrics['cv_score_std']:.3f})")

if __name__ == "__main__":
    trainer = ModelTrainer()
    results, vectorizer = trainer.train_models()
    print_results(results)