import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

# AI Chatbot Libraries
from groq import Groq
import re
import time

# Utility Libraries
import pickle
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LoanPredictionSystem:
    """
    Main class for the Loan Prediction System with AI Chatbot integration.
    """
    
    def __init__(self, groq_api_key=None):
        """
        Initialize the Loan Prediction System.

        Args:
            groq_api_key (str): API key for Groq AI chatbot (optional, will use env var if not provided)
        """
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.model_trained = False
        self.dataset = None
        self.preprocessing_steps = []

        # Initialize AI chatbot
        self.groq_client = None

        # Use provided API key or get from environment
        api_key = groq_api_key or os.getenv('GROQ_API_KEY')

        if api_key:
            try:
                self.groq_client = Groq(api_key=api_key)
                print("✅ AI Chatbot initialized successfully!")
            except Exception as e:
                print(f"❌ Failed to initialize AI Chatbot: {e}")
        else:
            print("⚠️ No Groq API key provided. AI Chatbot features will be disabled.")
            print("💡 Set GROQ_API_KEY environment variable to enable AI features.")
    
    def load_dataset(self, file_path):
        """
        Load the loan dataset from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.dataset = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Dataset shape: {self.dataset.shape}")
            print(f"📋 Columns: {list(self.dataset.columns)}")
            return self.dataset
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def analyze_data_quality(self, df=None):
        """
        Analyze data quality including missing values, outliers, and distributions.
        
        Args:
            df (pd.DataFrame): Dataset to analyze (uses self.dataset if None)
            
        Returns:
            dict: Data quality analysis results
        """
        if df is None:
            df = self.dataset
        
        if df is None:
            print("❌ No dataset available for analysis")
            return None
        
        print("\n" + "="*50)
        print("📊 DATA QUALITY ANALYSIS")
        print("="*50)
        
        # Basic info
        print(f"Dataset Shape: {df.shape}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values analysis
        print("\n📋 Missing Values Analysis:")
        missing_values = df.isnull().sum()
        missing_pct = (missing_values / len(df)) * 100
        
        missing_summary = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Count': missing_values.values,
            'Missing Percentage': missing_pct.values
        })
        missing_summary = missing_summary[missing_summary['Missing Count'] > 0]
        
        if not missing_summary.empty:
            print(missing_summary.to_string(index=False))
        else:
            print("✅ No missing values found!")
        
        # Data types
        print("\n📊 Data Types:")
        print(df.dtypes.value_counts())
        
        # Numerical columns analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\n🔢 Numerical Columns ({len(numerical_cols)}):")
            for col in numerical_cols:
                if col != 'Loan_ID':
                    outliers = self.detect_outliers_iqr(df, col)
                    print(f"  • {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\n📝 Categorical Columns ({len(categorical_cols)}):")
            for col in categorical_cols:
                if col not in ['Loan_ID']:
                    unique_count = df[col].nunique()
                    print(f"  • {col}: {unique_count} unique values")
        
        return {
            'missing_summary': missing_summary,
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'shape': df.shape
        }
    
    def detect_outliers_iqr(self, df, column):
        """
        Detect outliers using IQR method.
        
        Args:
            df (pd.DataFrame): Dataset
            column (str): Column name to analyze
            
        Returns:
            pd.DataFrame: Outliers detected
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers
    
    def handle_outliers(self, df, column, method='iqr', factor=1.5):
        """
        Handle outliers using different methods.
        
        Args:
            df (pd.DataFrame): Dataset
            column (str): Column name
            method (str): Method to handle outliers ('iqr', 'zscore')
            factor (float): Factor for IQR method
            
        Returns:
            pd.DataFrame: Dataset with outliers handled
        """
        data = df.copy()
        
        if method == 'iqr':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Cap outliers
            data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            data = data[z_scores < 3]  # Remove outliers beyond 3 standard deviations
        
        return data
    
    def advanced_missing_value_handling(self, df, strategy_dict=None):
        """
        Advanced missing value handling with different strategies for different columns.
        
        Args:
            df (pd.DataFrame): Dataset
            strategy_dict (dict): Dictionary mapping columns to strategies
            
        Returns:
            tuple: (processed_dataframe, missing_info)
        """
        data = df.copy()
        
        if strategy_dict is None:
            strategy_dict = {
                'Gender': 'mode',
                'Married': 'mode',
                'Dependents': 'mode',
                'Education': 'mode',
                'Self_Employed': 'mode',
                'ApplicantIncome': 'median',
                'CoapplicantIncome': 'median',
                'LoanAmount': 'median',
                'Loan_Amount_Term': 'mode',
                'Credit_History': 'mode',
                'Property_Area': 'mode'
            }
        
        missing_info = {}
        
        for column, strategy in strategy_dict.items():
            if column in data.columns and data[column].isnull().sum() > 0:
                missing_count = data[column].isnull().sum()
                missing_info[column] = {
                    'missing_count': missing_count,
                    'strategy': strategy,
                    'original_missing_pct': (missing_count / len(data)) * 100
                }
                
                if strategy == 'mode':
                    fill_value = data[column].mode()[0] if not data[column].mode().empty else 'Unknown'
                    data[column].fillna(fill_value, inplace=True)
                elif strategy == 'median':
                    fill_value = data[column].median()
                    data[column].fillna(fill_value, inplace=True)
                elif strategy == 'mean':
                    fill_value = data[column].mean()
                    data[column].fillna(fill_value, inplace=True)
        
        return data, missing_info
    
    def feature_engineering(self, df):
        """
        Create new features from existing ones.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            pd.DataFrame: Dataset with new features
        """
        data = df.copy()
        
        print("\n🔧 Feature Engineering:")
        
        # Total Income
        if 'ApplicantIncome' in data.columns and 'CoapplicantIncome' in data.columns:
            data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
            print("  ✅ Created TotalIncome feature")
        
        # Income to Loan Ratio
        if 'TotalIncome' in data.columns and 'LoanAmount' in data.columns:
            data['IncomeToLoanRatio'] = data['TotalIncome'] / (data['LoanAmount'] * 1000)
            data['IncomeToLoanRatio'] = data['IncomeToLoanRatio'].replace([np.inf, -np.inf], 0)
            print("  ✅ Created IncomeToLoanRatio feature")
        
        # Loan Amount per Term
        if 'LoanAmount' in data.columns and 'Loan_Amount_Term' in data.columns:
            data['LoanAmountPerTerm'] = data['LoanAmount'] / data['Loan_Amount_Term']
            data['LoanAmountPerTerm'] = data['LoanAmountPerTerm'].replace([np.inf, -np.inf], 0)
            print("  ✅ Created LoanAmountPerTerm feature")
        
        # Income Category
        if 'TotalIncome' in data.columns:
            data['IncomeCategory'] = pd.cut(data['TotalIncome'], 
                                           bins=[0, 3000, 6000, 10000, np.inf], 
                                           labels=['Low', 'Medium', 'High', 'Very High'])
            print("  ✅ Created IncomeCategory feature")
        
        return data

    def preprocess_data(self, df=None, handle_outliers_flag=True, feature_engineering_flag=True):
        """
        Enhanced preprocessing pipeline with comprehensive data cleaning.

        Args:
            df (pd.DataFrame): Dataset to preprocess (uses self.dataset if None)
            handle_outliers_flag (bool): Whether to handle outliers
            feature_engineering_flag (bool): Whether to perform feature engineering

        Returns:
            tuple: (X, y, label_encoders, feature_columns, preprocessing_steps)
        """
        if df is None:
            df = self.dataset

        if df is None:
            print("❌ No dataset available for preprocessing")
            return None

        print("\n" + "="*50)
        print("🔧 DATA PREPROCESSING")
        print("="*50)

        # Create a copy to avoid modifying original data
        data = df.copy()
        preprocessing_steps = []

        # Step 1: Handle missing values
        print("\n1️⃣ Handling missing values...")
        data, missing_info = self.advanced_missing_value_handling(data)
        preprocessing_steps.append(f"Handled missing values: {len(missing_info)} columns processed")

        # Step 2: Feature Engineering
        if feature_engineering_flag:
            print("\n2️⃣ Feature engineering...")
            original_cols = data.columns.tolist()
            data = self.feature_engineering(data)
            new_cols = [col for col in data.columns if col not in original_cols]
            if new_cols:
                preprocessing_steps.append(f"Created new features: {new_cols}")

        # Step 3: Handle outliers in numerical columns
        if handle_outliers_flag:
            print("\n3️⃣ Handling outliers...")
            numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
            for col in numerical_cols:
                if col in data.columns:
                    original_outliers = len(self.detect_outliers_iqr(data, col))
                    data = self.handle_outliers(data, col, method='iqr')
                    print(f"  ✅ Handled {original_outliers} outliers in {col}")
                    preprocessing_steps.append(f"Handled outliers in {col}")

        # Step 4: Convert Dependents to numeric
        if 'Dependents' in data.columns:
            print("\n4️⃣ Converting Dependents to numeric...")
            data['Dependents'] = data['Dependents'].replace('3+', '3')
            data['Dependents'] = pd.to_numeric(data['Dependents'])
            preprocessing_steps.append("Converted Dependents to numeric")

        # Step 5: Encode categorical variables
        print("\n5️⃣ Encoding categorical variables...")
        label_encoders = {}
        categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

        # Add new categorical columns from feature engineering
        if 'IncomeCategory' in data.columns:
            categorical_columns.append('IncomeCategory')

        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                label_encoders[col] = le
                print(f"  ✅ Encoded {col}")
                preprocessing_steps.append(f"Encoded categorical variable: {col}")

        # Step 6: Prepare features and target
        print("\n6️⃣ Preparing features and target...")
        feature_columns = [col for col in data.columns if col not in ['Loan_ID', 'Loan_Status']]
        X = data[feature_columns]
        y = LabelEncoder().fit_transform(data['Loan_Status'])

        print(f"  ✅ Features shape: {X.shape}")
        print(f"  ✅ Target distribution: {np.bincount(y)}")

        # Store for later use
        self.label_encoders = label_encoders
        self.feature_columns = feature_columns
        self.preprocessing_steps = preprocessing_steps

        return X, y, label_encoders, feature_columns, preprocessing_steps

    def create_deep_learning_model(self, input_shape):
        """
        Create an improved deep learning model with regularization to prevent overfitting.

        Args:
            input_shape (int): Number of input features

        Returns:
            tf.keras.Model: Compiled model
        """
        print("\n🧠 Creating Deep Learning Model...")

        model = Sequential([
            # Input layer with L1/L2 regularization
            Dense(64, activation='relu', input_shape=(input_shape,),
                  kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.4),

            # Hidden layer 1
            Dense(32, activation='relu',
                  kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.3),

            # Hidden layer 2
            Dense(16, activation='relu',
                  kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.2),

            # Output layer
            Dense(1, activation='sigmoid')
        ])

        # Use a lower learning rate to prevent overfitting
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print("  ✅ Model architecture created")
        print(f"  📊 Total parameters: {model.count_params():,}")

        return model

    def train_model(self, X=None, y=None, test_size=0.2, epochs=200, batch_size=16):
        """
        Train the deep learning model with enhanced metrics and overfitting prevention.

        Args:
            X (pd.DataFrame): Features (uses preprocessed data if None)
            y (np.array): Target (uses preprocessed data if None)
            test_size (float): Test set size
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size for training

        Returns:
            dict: Training results and metrics
        """
        print("\n" + "="*50)
        print("🚀 MODEL TRAINING")
        print("="*50)

        if X is None or y is None:
            print("❌ No preprocessed data available. Please run preprocess_data() first.")
            return None

        # Split the data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"📊 Training set: {X_train.shape}")
        print(f"📊 Test set: {X_test.shape}")

        # Scale the features
        print("\n🔄 Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create the improved model
        self.model = self.create_deep_learning_model(X_train_scaled.shape[1])

        # Define callbacks to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            verbose=1
        )

        callbacks = [early_stopping, reduce_lr]

        print(f"\n🏋️ Training model for up to {epochs} epochs...")
        print("⏰ Early stopping enabled to prevent overfitting")

        # Train the model
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate the model
        print("\n📊 Evaluating model performance...")
        train_accuracy = self.model.evaluate(X_train_scaled, y_train, verbose=0)[1]
        test_accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)[1]

        # Calculate additional metrics
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Classification report and confusion matrix
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        confusion_mat = confusion_matrix(y_test, y_pred)

        # ROC AUC Score
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Calculate feature importance using Random Forest for interpretation
        print("\n🎯 Calculating feature importance...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        feature_importance = rf_model.feature_importances_

        # Training results
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'roc_auc': roc_auc,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat,
            'feature_importance': feature_importance,
            'feature_names': X.columns.tolist(),
            'history': history,
            'epochs_trained': len(history.history['accuracy'])
        }

        # Print results
        print(f"\n✅ Training completed!")
        print(f"📈 Training Accuracy: {train_accuracy:.4f}")
        print(f"📈 Test Accuracy: {test_accuracy:.4f}")
        print(f"📈 ROC AUC Score: {roc_auc:.4f}")
        print(f"🔄 Epochs Trained: {results['epochs_trained']}")

        # Check for overfitting
        overfitting_gap = train_accuracy - test_accuracy
        if overfitting_gap < 0.05:
            print("✅ Good generalization - No significant overfitting detected")
        elif overfitting_gap < 0.1:
            print("⚠️ Slight overfitting detected")
        else:
            print("❌ Significant overfitting detected")

        self.model_trained = True
        return results

    def predict_single_loan(self, user_input):
        """
        Predict loan approval for a single applicant.

        Args:
            user_input (dict): Dictionary containing applicant information

        Returns:
            tuple: (prediction, probability)
        """
        if not self.model_trained:
            print("❌ Model not trained. Please train the model first.")
            return None, None

        # Create a dataframe with user input
        input_df = pd.DataFrame([user_input])

        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except ValueError:
                    # Handle unseen categories
                    input_df[col] = 0

        # Reorder columns to match training data
        input_df = input_df[self.feature_columns]

        # Scale the input
        input_scaled = self.scaler.transform(input_df)

        # Make prediction
        prediction = self.model.predict(input_scaled)[0][0]
        probability = prediction

        return "Approved" if prediction > 0.5 else "Not Approved", probability

    def predict_bulk_loans(self, test_df):
        """
        Predict loan approval for multiple applicants.

        Args:
            test_df (pd.DataFrame): DataFrame containing multiple loan applications

        Returns:
            pd.DataFrame: DataFrame with predictions and probabilities
        """
        if not self.model_trained:
            print("❌ Model not trained. Please train the model first.")
            return None

        print(f"\n🔮 Making bulk predictions for {len(test_df)} applications...")

        # Create a copy to avoid modifying original data
        data = test_df.copy()
        original_data = test_df.copy()

        # Preprocess the test data similar to training data
        data, _ = self.advanced_missing_value_handling(data)

        # Feature engineering (if it was used during training)
        if 'TotalIncome' in self.feature_columns:
            data = self.feature_engineering(data)

        # Convert Dependents to numeric
        if 'Dependents' in data.columns:
            data['Dependents'] = data['Dependents'].replace('3+', '3')
            data['Dependents'] = pd.to_numeric(data['Dependents'], errors='coerce').fillna(0)

        # Encode categorical variables using the same encoders from training
        for col, encoder in self.label_encoders.items():
            if col in data.columns:
                try:
                    # Handle unseen categories by mapping them to the most frequent category
                    known_values = encoder.classes_
                    most_frequent = encoder.classes_[0]
                    data[col] = data[col].apply(lambda x: x if x in known_values else most_frequent)
                    data[col] = encoder.transform(data[col])
                except Exception:
                    # If encoding fails, fill with 0
                    data[col] = 0

        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0

        # Reorder columns to match training data
        data = data[self.feature_columns]

        # Scale the features
        data_scaled = self.scaler.transform(data)

        # Make predictions
        predictions_proba = self.model.predict(data_scaled)
        predictions = (predictions_proba > 0.5).astype(int)

        # Create results dataframe
        results_df = original_data.copy()
        results_df['Prediction_Probability'] = predictions_proba.flatten()
        results_df['Prediction'] = ['Approved' if pred == 1 else 'Not Approved' for pred in predictions]
        results_df['Confidence'] = results_df['Prediction_Probability'].apply(
            lambda x: x if x > 0.5 else 1 - x
        )

        print(f"✅ Bulk predictions completed!")
        print(f"📊 Approved: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
        print(f"📊 Average Confidence: {results_df['Confidence'].mean():.3f}")

        return results_df

    def generate_ai_response(self, user_message, context=""):
        """
        Generate response using Groq AI chatbot.

        Args:
            user_message (str): User's message
            context (str): Additional context for the AI

        Returns:
            str: AI response
        """
        if not self.groq_client:
            return "❌ AI Chatbot not initialized. Please provide a valid API key."

        try:
            system_prompt = f"""You are a helpful loan advisor assistant. You help users understand loan approval processes and gather necessary information for loan applications.

Context about the loan application system:
- The system predicts loan approval based on factors like income, credit history, education, employment status, etc.
- Required information: Gender, Marital Status, Dependents, Education, Employment Status, Applicant Income, Coapplicant Income, Loan Amount, Loan Term, Credit History, Property Area

Current context: {context}

Please provide helpful, accurate, and friendly responses about loan applications. If asked about specific loan details, guide the user through the application process step by step."""

            completion = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            return completion.choices[0].message.content
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def extract_loan_details(self, text):
        """
        Extract loan application details from user text using regex.

        Args:
            text (str): User input text

        Returns:
            dict: Extracted loan details
        """
        details = {}

        # Income extraction
        income_match = re.search(r'income.*?(\d+)', text.lower())
        if income_match:
            details['ApplicantIncome'] = int(income_match.group(1))

        # Loan amount extraction
        loan_match = re.search(r'loan.*?amount.*?(\d+)', text.lower())
        if loan_match:
            details['LoanAmount'] = int(loan_match.group(1))

        # Gender extraction
        if 'male' in text.lower() and 'female' not in text.lower():
            details['Gender'] = 'Male'
        elif 'female' in text.lower():
            details['Gender'] = 'Female'

        # Marital status
        if 'married' in text.lower():
            details['Married'] = 'Yes'
        elif 'single' in text.lower() or 'unmarried' in text.lower():
            details['Married'] = 'No'

        # Education
        if 'graduate' in text.lower():
            details['Education'] = 'Graduate'
        elif 'not graduate' in text.lower():
            details['Education'] = 'Not Graduate'

        return details

    def visualize_training_results(self, results):
        """
        Create visualizations for training results.

        Args:
            results (dict): Training results from train_model()
        """
        if not results:
            print("❌ No training results available for visualization")
            return

        print("\n📊 Creating visualizations...")

        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Loan Prediction Model - Training Results', fontsize=16, fontweight='bold')

        # 1. Training History - Accuracy
        history = results['history']
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Training History - Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Feature Importance
        feature_importance = results['feature_importance']
        feature_names = results['feature_names']

        # Get top 10 features
        top_indices = np.argsort(feature_importance)[-10:]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = feature_importance[top_indices]

        axes[1, 0].barh(range(len(top_features)), top_importance)
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features)
        axes[1, 0].set_title('Top 10 Feature Importance')
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Confusion Matrix
        cm = results['confusion_matrix']
        im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 1].set_title('Confusion Matrix')

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")

        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_xlabel('Predicted Label')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_xticklabels(['Not Approved', 'Approved'])
        axes[1, 1].set_yticklabels(['Not Approved', 'Approved'])

        plt.tight_layout()
        plt.show()

        # Print detailed metrics
        print("\n📈 DETAILED PERFORMANCE METRICS")
        print("="*50)
        print(f"Training Accuracy: {results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"ROC AUC Score: {results['roc_auc']:.4f}")
        print(f"Epochs Trained: {results['epochs_trained']}")

        # Classification report
        print(f"\n📊 Classification Report:")
        class_report = results['classification_report']
        print(f"Precision (Not Approved): {class_report['0']['precision']:.3f}")
        print(f"Recall (Not Approved): {class_report['0']['recall']:.3f}")
        print(f"F1-Score (Not Approved): {class_report['0']['f1-score']:.3f}")
        print(f"Precision (Approved): {class_report['1']['precision']:.3f}")
        print(f"Recall (Approved): {class_report['1']['recall']:.3f}")
        print(f"F1-Score (Approved): {class_report['1']['f1-score']:.3f}")

    def save_model(self, filepath):
        """
        Save the trained model and preprocessing components.

        Args:
            filepath (str): Path to save the model
        """
        if not self.model_trained:
            print("❌ No trained model to save")
            return

        try:
            # Save model
            self.model.save(f"{filepath}_model.h5")

            # Save preprocessing components
            model_components = {
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'preprocessing_steps': self.preprocessing_steps
            }

            with open(f"{filepath}_components.pkl", 'wb') as f:
                pickle.dump(model_components, f)

            print(f"✅ Model saved successfully to {filepath}")

        except Exception as e:
            print(f"❌ Error saving model: {e}")

    def load_model(self, filepath):
        """
        Load a previously trained model and preprocessing components.

        Args:
            filepath (str): Path to load the model from
        """
        try:
            # Load model
            self.model = tf.keras.models.load_model(f"{filepath}_model.h5")

            # Load preprocessing components
            with open(f"{filepath}_components.pkl", 'rb') as f:
                components = pickle.load(f)

            self.scaler = components['scaler']
            self.label_encoders = components['label_encoders']
            self.feature_columns = components['feature_columns']
            self.preprocessing_steps = components['preprocessing_steps']

            self.model_trained = True
            print(f"✅ Model loaded successfully from {filepath}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")


def demonstrate_system():
    """
    Demonstrate the complete loan prediction system functionality.
    """
    print("="*70)
    print("🏦 LOAN PREDICTION & AI CHATBOT SYSTEM DEMONSTRATION")
    print("="*70)

    # Initialize the system
    print("\n🚀 Initializing Loan Prediction System...")

    # Initialize system (will automatically use GROQ_API_KEY from environment)
    system = LoanPredictionSystem()

    # Load dataset
    print("\n📂 Loading dataset...")
    dataset_path = "Dataset/Training/Training Dataset.csv"

    try:
        dataset = system.load_dataset(dataset_path)

        if dataset is not None:
            # Analyze data quality
            system.analyze_data_quality()

            # Preprocess data
            print("\n🔧 Preprocessing data...")
            X, y, _, _, _ = system.preprocess_data()

            # Train model
            print("\n🏋️ Training model...")
            results = system.train_model(X, y, epochs=50)  # Reduced epochs for demo

            if results:
                # Visualize results
                system.visualize_training_results(results)

                # Demonstrate single prediction
                print("\n🔮 Demonstrating single loan prediction...")
                sample_application = {
                    'Gender': 'Male',
                    'Married': 'Yes',
                    'Dependents': '0',
                    'Education': 'Graduate',
                    'Self_Employed': 'No',
                    'ApplicantIncome': 5849,
                    'CoapplicantIncome': 0,
                    'LoanAmount': 128,
                    'Loan_Amount_Term': 360,
                    'Credit_History': 1,
                    'Property_Area': 'Urban'
                }

                prediction, probability = system.predict_single_loan(sample_application)
                print(f"📊 Sample Application Result:")
                print(f"   Prediction: {prediction}")
                print(f"   Confidence: {probability:.3f}")

                # Demonstrate AI chatbot
                print("\n🤖 Demonstrating AI Chatbot...")
                user_messages = [
                    "Hi, I want to apply for a loan. What information do I need?",
                    "I am a 30-year-old male, married, with an income of 50000. Can I get a loan?",
                    "What factors affect loan approval?"
                ]

                for message in user_messages:
                    print(f"\n👤 User: {message}")
                    response = system.generate_ai_response(message)
                    print(f"🤖 AI: {response}")

                    # Extract details if any
                    extracted = system.extract_loan_details(message)
                    if extracted:
                        print(f"📋 Extracted Details: {extracted}")

                # Save model
                print("\n💾 Saving trained model...")
                system.save_model("trained_loan_model")

                print("\n✅ Demonstration completed successfully!")
                print("\n📋 System Features Demonstrated:")
                print("   ✅ Data loading and quality analysis")
                print("   ✅ Advanced data preprocessing")
                print("   ✅ Feature engineering")
                print("   ✅ Model training with overfitting prevention")
                print("   ✅ Comprehensive evaluation and visualization")
                print("   ✅ Single loan prediction")
                print("   ✅ AI chatbot integration")
                print("   ✅ Model saving and loading")

            else:
                print("❌ Model training failed")

    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_path}")
        print("📝 Please ensure the dataset file exists in the correct location")

        # Create sample data for demonstration
        print("\n🔧 Creating sample dataset for demonstration...")
        sample_data = create_sample_dataset()

        if sample_data is not None:
            system.dataset = sample_data
            print("✅ Sample dataset created successfully!")

            # Continue with demonstration using sample data
            system.analyze_data_quality()
            X, y, _, _, _ = system.preprocess_data()
            results = system.train_model(X, y, epochs=20)

            if results:
                system.visualize_training_results(results)


def create_sample_dataset():
    """
    Create a sample dataset for demonstration purposes.

    Returns:
        pd.DataFrame: Sample loan dataset
    """
    np.random.seed(42)

    n_samples = 500

    # Generate sample data
    data = {
        'Loan_ID': [f'LP{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.7, 0.3]),
        'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.7, 0.3]),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
        'ApplicantIncome': np.random.normal(5000, 2000, n_samples).astype(int),
        'CoapplicantIncome': np.random.normal(1500, 1000, n_samples).astype(int),
        'LoanAmount': np.random.normal(150, 50, n_samples).astype(int),
        'Loan_Amount_Term': np.random.choice([120, 240, 360, 480], n_samples, p=[0.1, 0.2, 0.6, 0.1]),
        'Credit_History': np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),
        'Property_Area': np.random.choice(['Urban', 'Rural', 'Semiurban'], n_samples, p=[0.4, 0.3, 0.3])
    }

    # Create target variable based on some logic
    loan_status = []
    for i in range(n_samples):
        # Simple logic for loan approval
        score = 0
        if data['ApplicantIncome'][i] > 4000:
            score += 1
        if data['Credit_History'][i] == 1:
            score += 2
        if data['Education'][i] == 'Graduate':
            score += 1
        if data['LoanAmount'][i] < 200:
            score += 1

        # Add some randomness
        if np.random.random() < 0.1:
            score = np.random.randint(0, 5)

        loan_status.append('Y' if score >= 3 else 'N')

    data['Loan_Status'] = loan_status

    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    for idx in missing_indices[:20]:
        data['Gender'][idx] = None
    for idx in missing_indices[20:40]:
        data['LoanAmount'][idx] = None

    df = pd.DataFrame(data)

    # Ensure positive values
    df['ApplicantIncome'] = df['ApplicantIncome'].clip(lower=1000)
    df['CoapplicantIncome'] = df['CoapplicantIncome'].clip(lower=0)
    df['LoanAmount'] = df['LoanAmount'].clip(lower=10)

    return df


if __name__ == "__main__":
    """
    Main execution block - Run the complete demonstration.
    """
    try:
        demonstrate_system()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred during demonstration: {e}")
        print("📝 Please check your environment setup and try again")

    print("\n" + "="*70)
    print("🎯 LOAN PREDICTION & AI CHATBOT SYSTEM")
    print("📧 For questions or support, please contact the development team")
    print("🔗 This system demonstrates advanced ML techniques for loan approval")
    print("="*70)
