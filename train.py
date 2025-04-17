import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class MentalHealthDetector:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Reduce max_features for more lightweight model
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.traditional_model = None
        
    def preprocess_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^a-zA-Z]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            tokens = nltk.word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
            return ' '.join(tokens)
        return ""
    
    def load_dataset1(self, file_path):
        df = pd.read_csv(file_path)
        df['clean_statement'] = df['statement'].apply(self.preprocess_text)
        df['binary_label'] = df['status'].apply(lambda x: 0 if x == 'Normal' else 1)
        return df
    
    def load_dataset2(self, file_path):
        df = pd.read_excel(file_path, engine='openpyxl')
        df = df.dropna(subset=['label'])
        df['clean_text'] = df['text'].apply(self.preprocess_text)
        df['label'] = df['label'].astype(int)
        return df
    
    def load_dataset3(self, file_path):
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['label'])
        df['clean_text'] = df['text'].apply(self.preprocess_text)
        df['label'] = df['label'].astype(int)
        return df
    
    def combine_text_datasets(self, datasets):
        combined_texts = []
        combined_labels = []
        for df, text_col, label_col in datasets:
            combined_texts.extend(df[text_col].tolist())
            combined_labels.extend(df[label_col].tolist())
        return combined_texts, combined_labels
    
    def build_traditional_model(self, X_train_tfidf, y_train):
        # Using lighter models and fewer of them
        log_reg = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='liblinear')
        
        # LinearSVC is much faster than SVC with rbf kernel
        # Wrap in CalibratedClassifierCV to get probability estimates
        linear_svc = CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight='balanced', max_iter=1000, dual=False)
        )
        
        # Create lightweight voting classifier with just 2 models
        voting_clf = VotingClassifier(
            estimators=[('lr', log_reg), ('svc', linear_svc)],
            voting='soft'
        )
        
        print("Training lightweight ensemble classifier...")
        self.traditional_model = voting_clf.fit(X_train_tfidf, y_train)
        return self.traditional_model
    
    def build_single_model(self, X_train_tfidf, y_train):
        # Just use logistic regression - very lightweight and interpretable
        log_reg = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='liblinear')
        print("Training single logistic regression model...")
        self.traditional_model = log_reg.fit(X_train_tfidf, y_train)
        return self.traditional_model
    
    def find_best_model(self, X_train_tfidf, y_train):
        """Use GridSearchCV to find the best performing lightweight model"""
        print("Finding best lightweight model with GridSearchCV...")
        
        # Define lightweight models to test
        models = {
            'logistic': {
                'model': LogisticRegression(class_weight='balanced'),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear'],
                    'max_iter': [1000]
                }
            },
            'linear_svc': {
                'model': CalibratedClassifierCV(LinearSVC(class_weight='balanced', dual=False)),
                'params': {
                    'base_estimator__C': [0.1, 1.0, 10.0],
                    'base_estimator__max_iter': [1000]
                }
            }
        }
        
        best_score = 0
        best_model = None
        
        # Sample a subset of data for faster grid search if dataset is large
        sample_size = min(5000, X_train_tfidf.shape[0])
        indices = np.random.choice(X_train_tfidf.shape[0], sample_size, replace=False)
        X_sample = X_train_tfidf[indices]
        y_sample = np.array(y_train)[indices]
        
        for model_name, model_info in models.items():
            print(f"Evaluating {model_name}...")
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=3,
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(X_sample, y_sample)
            
            print(f"Best {model_name} score: {grid_search.best_score_:.4f}")
            print(f"Best parameters: {grid_search.best_params_}")
            
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
        
        print(f"Best overall model found with score: {best_score:.4f}")
        return best_model
    
    def train_and_evaluate(self, datasets_paths, model_type='single', use_grid_search=False):
        """
        Train and evaluate the model
        model_type: 'single', 'ensemble', or 'best'
        """
        print("Loading and preprocessing datasets...")
        try:
            dataset1 = self.load_dataset1(datasets_paths['dataset1'])
            dataset2 = self.load_dataset2(datasets_paths['dataset2'])
            dataset3 = self.load_dataset3(datasets_paths['dataset3'])
            
            print("Datasets loaded successfully.")
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            return
        
        datasets = [
            (dataset1, 'clean_statement', 'binary_label'),
            (dataset2, 'clean_text', 'label'),
            (dataset3, 'clean_text', 'label')
        ]
        combined_texts, combined_labels = self.combine_text_datasets(datasets)
        
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            combined_texts, combined_labels, test_size=0.2, random_state=42, stratify=combined_labels
        )
        
        print(f"Training data: {len(X_train_text)} samples")
        print(f"Testing data: {len(X_test_text)} samples")
        
        self.tfidf_vectorizer.fit(X_train_text)
        X_train_tfidf = self.tfidf_vectorizer.transform(X_train_text)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test_text)
        
        print("\nTraining models...")
        if use_grid_search:
            # Find best single model with grid search
            self.traditional_model = self.find_best_model(X_train_tfidf, y_train)
        elif model_type == 'ensemble':
            # Use ensemble voting classifier (still lightweight)
            self.build_traditional_model(X_train_tfidf, y_train)
        else:  # Default to single model
            # Use single logistic regression (most lightweight)
            self.build_single_model(X_train_tfidf, y_train)
        
        y_pred_trad = self.traditional_model.predict(X_test_tfidf)
        y_pred_proba_trad = self.traditional_model.predict_proba(X_test_tfidf)[:, 1]
        
        # Plot ROC curve to help determine the optimal threshold
        thresholds = self.plot_roc_curve(y_test, y_pred_proba_trad)
        
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_trad):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred_trad):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_trad):.4f}")
        print(classification_report(y_test, y_pred_trad))
        
        self.plot_confusion_matrix(y_test, y_pred_trad)
        
        # Analyze the most important features for interpretation
        self.analyze_important_features()
        
        return {
            'traditional_model': self.traditional_model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'metrics': {
                'traditional': {
                    'accuracy': accuracy_score(y_test, y_pred_trad),
                    'f1': f1_score(y_test, y_pred_trad),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba_trad)
                }
            }
        }
    
    def analyze_important_features(self):
        """Analyze and print the most important features if possible"""
        if not hasattr(self, 'tfidf_vectorizer') or not self.tfidf_vectorizer:
            print("TF-IDF vectorizer not available for feature analysis")
            return
            
        if hasattr(self.traditional_model, 'named_steps') and 'lr' in self.traditional_model.named_steps:
            # For pipeline with logistic regression
            model = self.traditional_model.named_steps['lr']
            self._print_important_features(model)
        elif isinstance(self.traditional_model, LogisticRegression):
            # For direct logistic regression
            self._print_important_features(self.traditional_model)
        elif hasattr(self.traditional_model, 'estimators_') and len(self.traditional_model.estimators_) > 0:
            # For voting classifier, try to access logistic regression if present
            for est_name, estimator in self.traditional_model.named_estimators_.items():
                if isinstance(estimator, LogisticRegression):
                    print(f"\nImportant features from {est_name}:")
                    self._print_important_features(estimator)
                    break
        else:
            print("Model type doesn't support direct feature importance extraction")
    
    def _print_important_features(self, model):
        """Helper method to print important features for linear models"""
        try:
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            if hasattr(model, 'coef_') and model.coef_ is not None:
                # For binary classification
                coefficients = model.coef_[0]
                # Get top positive and negative features
                top_positive_idx = np.argsort(coefficients)[-10:]  # Reduced from 20 to 10
                top_negative_idx = np.argsort(coefficients)[:10]   # Reduced from 20 to 10
                
                print("\nTop positive features (indicating mental health issues):")
                for idx in reversed(top_positive_idx):
                    print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")
                    
                print("\nTop negative features (indicating normal mental state):")
                for idx in top_negative_idx:
                    print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")
        except Exception as e:
            print(f"Error extracting feature importance: {str(e)}")
    
    def predict(self, text, threshold=0.5):
        """
        Predict the mental health status of a given text.
        Allows for a customizable threshold.
        """
        clean_text = self.preprocess_text(text)
        text_tfidf = self.tfidf_vectorizer.transform([clean_text])
        
        trad_pred_prob = self.traditional_model.predict_proba(text_tfidf)[0, 1]
        trad_pred_class = 1 if trad_pred_prob > threshold else 0
        
        # Risk level determination
        if trad_pred_prob < 0.3:
            risk_level = "Low Risk"
        elif trad_pred_prob < 0.5:
            risk_level = "Low-Moderate Risk"
        elif trad_pred_prob < 0.7:
            risk_level = "Moderate Risk"
        elif trad_pred_prob < 0.85:
            risk_level = "Moderate-High Risk"
        else:
            risk_level = "High Risk"
        
        # More detailed condition detection
        condition = None
        if trad_pred_class == 1:
            if trad_pred_prob > 0.9:
                condition = "Severe mental health concern detected"
            elif trad_pred_prob > 0.7:
                condition = "Significant mental health concern detected"
            else:
                condition = "Potential mental health concern detected"
                
            # The model should suggest next steps based on risk
            if trad_pred_prob > 0.7:
                recommendation = "Consider professional mental health support"
            else:
                recommendation = "Consider speaking with a trusted individual about your feelings"
        else:
            recommendation = "No specific mental health recommendations at this time"
        
        return {
            'prediction': trad_pred_class,
            'probability': trad_pred_prob,
            'risk_level': risk_level,
            'condition': condition,
            'recommendation': recommendation
        }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Mental Health Issue'],
                   yticklabels=['Normal', 'Mental Health Issue'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_curve(self, y_true, y_proba):
        """Plot the ROC curve to help determine the optimal threshold."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()
        return thresholds

    def save_models(self, path_prefix):
        import joblib
        import os
        
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        joblib.dump(self.traditional_model, f"{path_prefix}_traditional_model.joblib")
        joblib.dump(self.tfidf_vectorizer, f"{path_prefix}_tfidf_vectorizer.joblib")
        print(f"Models saved with prefix: {path_prefix}")
    
    def load_models(self, path_prefix):
        import joblib
        
        self.traditional_model = joblib.load(f"{path_prefix}_traditional_model.joblib")
        self.tfidf_vectorizer = joblib.load(f"{path_prefix}_tfidf_vectorizer.joblib")
        print(f"Models loaded from prefix: {path_prefix}")

if __name__ == "__main__":
    mh_detector = MentalHealthDetector()
    datasets_paths = {
        'dataset1': "/kaggle/input/sentiment-analysis-for-mental-health/Combined Data.csv",
        'dataset2': "/kaggle/input/student-depression-text/Depression_Text.xlsx",
        'dataset3': "/kaggle/input/mental-health-corpus/mental_health.csv"
    }
    
    # Choose model type:
    # 'single' - just logistic regression (most lightweight)
    # 'ensemble' - LogisticRegression + LinearSVC (still lightweight)
    # If use_grid_search=True, will find the best model through grid search
    results = mh_detector.train_and_evaluate(datasets_paths, model_type='single', use_grid_search=False)
    mh_detector.save_models("models/mental_health")
    
    # Test with sample texts
    sample_texts = [
        "I've been feeling extremely sad and hopeless for the past month.",
        "I'm having a great time with my friends and enjoying life.",
        "I can't sleep at night and have constant worries about everything.",
        "Today was productive, I accomplished all my tasks and feel satisfied."
    ]
    
    print("\nSample Predictions:")
    for text in sample_texts:
        prediction = mh_detector.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {prediction['prediction']} (1 = Mental health issue, 0 = Normal)")
        print(f"Probability: {prediction['probability']:.4f}")
        print(f"Risk Level: {prediction['risk_level']}")
        print(f"Condition: {prediction['condition']}")
        print(f"Recommendation: {prediction['recommendation']}")