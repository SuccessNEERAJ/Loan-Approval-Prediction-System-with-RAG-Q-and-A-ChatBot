import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import pickle
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq
import time
import re
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

# Groq client initialization
@st.cache_resource
def initialize_groq_client():
    """
    Initialize Groq client using environment variable for API key.
    """
    try:
        # Get API key from environment variable
        api_key = os.getenv('GROQ_API_KEY')

        if not api_key:
            st.error("❌ Groq API key not found! Please set GROQ_API_KEY in your .env file.")
            st.info("💡 Copy .env.example to .env and add your Groq API key.")
            return None

        client = Groq(api_key=api_key)
        st.success("✅ AI Chatbot initialized successfully!")
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {e}")
        st.info("💡 Please check your API key and internet connection.")
        return None

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def create_missing_values_heatmap(df):
    """Create a heatmap showing missing values pattern"""
    missing_data = df.isnull()
    fig = px.imshow(missing_data.values,
                    labels=dict(x="Columns", y="Rows", color="Missing"),
                    x=missing_data.columns,
                    color_continuous_scale=['lightblue', 'red'],
                    title="Missing Values Heatmap")
    fig.update_layout(height=400)
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap for numerical columns"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        fig = px.imshow(corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Correlation Heatmap",
                        color_continuous_scale='RdBu_r')
        fig.update_layout(height=500)
        return fig
    return None

def create_distribution_plots(df):
    """Create distribution plots for numerical columns"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    plots = {}

    # Numerical distributions
    for col in numerical_cols:
        if col != 'Loan_ID':
            fig = px.histogram(df, x=col, nbins=30, title=f'Distribution of {col}')
            fig.add_vline(x=df[col].mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {df[col].mean():.2f}")
            plots[f'{col}_dist'] = fig

    # Categorical distributions
    for col in categorical_cols:
        if col not in ['Loan_ID']:
            fig = px.bar(df[col].value_counts().reset_index(),
                        x='index', y=col, title=f'Distribution of {col}')
            plots[f'{col}_dist'] = fig

    return plots

def create_outlier_plots(df):
    """Create box plots to visualize outliers"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    plots = {}

    for col in numerical_cols:
        if col != 'Loan_ID':
            fig = px.box(df, y=col, title=f'Outliers in {col}')
            plots[f'{col}_outliers'] = fig

    return plots

def handle_outliers(df, column, method='iqr', factor=1.5):
    """Handle outliers using different methods"""
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

def advanced_missing_value_handling(df, strategy_dict=None):
    """Advanced missing value handling with different strategies for different columns"""
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
            elif strategy == 'forward_fill':
                data[column].fillna(method='ffill', inplace=True)
            elif strategy == 'backward_fill':
                data[column].fillna(method='bfill', inplace=True)

    return data, missing_info

def feature_engineering(df):
    """Create new features from existing ones"""
    data = df.copy()

    # Total Income
    if 'ApplicantIncome' in data.columns and 'CoapplicantIncome' in data.columns:
        data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

    # Income to Loan Ratio
    if 'TotalIncome' in data.columns and 'LoanAmount' in data.columns:
        data['IncomeToLoanRatio'] = data['TotalIncome'] / (data['LoanAmount'] * 1000)  # LoanAmount is in thousands
        data['IncomeToLoanRatio'] = data['IncomeToLoanRatio'].replace([np.inf, -np.inf], 0)

    # Loan Amount per Term
    if 'LoanAmount' in data.columns and 'Loan_Amount_Term' in data.columns:
        data['LoanAmountPerTerm'] = data['LoanAmount'] / data['Loan_Amount_Term']
        data['LoanAmountPerTerm'] = data['LoanAmountPerTerm'].replace([np.inf, -np.inf], 0)

    # Income Category
    if 'TotalIncome' in data.columns:
        data['IncomeCategory'] = pd.cut(data['TotalIncome'],
                                       bins=[0, 3000, 6000, 10000, np.inf],
                                       labels=['Low', 'Medium', 'High', 'Very High'])

    return data

def preprocess_data(df, handle_outliers_flag=True, feature_engineering_flag=True):
    """Enhanced preprocessing pipeline with comprehensive data cleaning"""
    # Create a copy to avoid modifying original data
    data = df.copy()

    preprocessing_steps = []

    # Step 1: Handle missing values
    data, missing_info = advanced_missing_value_handling(data)
    preprocessing_steps.append(f"Handled missing values: {missing_info}")

    # Step 2: Feature Engineering
    if feature_engineering_flag:
        original_cols = data.columns.tolist()
        data = feature_engineering(data)
        new_cols = [col for col in data.columns if col not in original_cols]
        if new_cols:
            preprocessing_steps.append(f"Created new features: {new_cols}")

    # Step 3: Handle outliers in numerical columns
    if handle_outliers_flag:
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
        for col in numerical_cols:
            if col in data.columns:
                data = handle_outliers(data, col, method='iqr')
                preprocessing_steps.append(f"Handled outliers in {col}")

    # Step 4: Convert Dependents to numeric
    if 'Dependents' in data.columns:
        data['Dependents'] = data['Dependents'].replace('3+', '3')
        data['Dependents'] = pd.to_numeric(data['Dependents'])
        preprocessing_steps.append("Converted Dependents to numeric")

    # Step 5: Encode categorical variables
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
            preprocessing_steps.append(f"Encoded categorical variable: {col}")

    # Step 6: Prepare features and target
    feature_columns = [col for col in data.columns if col not in ['Loan_ID', 'Loan_Status']]
    X = data[feature_columns]
    y = LabelEncoder().fit_transform(data['Loan_Status'])

    return X, y, label_encoders, feature_columns, preprocessing_steps

def create_deep_learning_model(input_shape):
    """Create an improved deep learning model with regularization to prevent overfitting"""
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

    return model

def train_model(X, y):
    """Train the deep learning model with enhanced metrics and overfitting prevention"""
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create the improved model
    model = create_deep_learning_model(X_train_scaled.shape[1])

    # Define callbacks to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.0001,
        verbose=0
    )

    callbacks = [early_stopping, reduce_lr]

    with st.spinner('Training the model with overfitting prevention...'):
        # Train with more epochs but early stopping will prevent overfitting
        history = model.fit(
            X_train_scaled, y_train,
            epochs=200,  # Increased epochs, but early stopping will prevent overfitting
            batch_size=16,  # Smaller batch size for better generalization
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )

    # Evaluate the model
    train_accuracy = model.evaluate(X_train_scaled, y_train, verbose=0)[1]
    test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)[1]

    # Calculate feature importance using Random Forest for interpretation
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    feature_importance = rf_model.feature_importances_

    # Calculate additional metrics
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    confusion_mat = confusion_matrix(y_test, y_pred)

    return (model, scaler, train_accuracy, test_accuracy, history,
            feature_importance, X.columns, classification_rep, confusion_mat)

def predict_loan_approval(model, scaler, label_encoders, feature_columns, user_input):
    """Predict loan approval for given user input"""
    # Create a dataframe with user input
    input_df = pd.DataFrame([user_input])

    # Ensure all required columns are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col])
            except ValueError:
                # Handle unseen categories
                input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df[feature_columns]

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0][0]
    probability = prediction

    return "Approved" if prediction > 0.5 else "Not Approved", probability

def predict_bulk_loan_approval(model, scaler, label_encoders, feature_columns, test_df):
    """Predict loan approval for bulk data"""
    # Create a copy to avoid modifying original data
    data = test_df.copy()

    # Store original data for results
    original_data = test_df.copy()

    # Preprocess the test data similar to training data
    # Handle missing values using the same strategy as training
    data, _ = advanced_missing_value_handling(data)

    # Feature engineering (if it was used during training)
    if 'TotalIncome' in feature_columns:
        data = feature_engineering(data)

    # Convert Dependents to numeric
    if 'Dependents' in data.columns:
        data['Dependents'] = data['Dependents'].replace('3+', '3')
        data['Dependents'] = pd.to_numeric(data['Dependents'], errors='coerce').fillna(0)

    # Encode categorical variables using the same encoders from training
    for col, encoder in label_encoders.items():
        if col in data.columns:
            try:
                # Handle unseen categories by mapping them to the most frequent category
                known_values = encoder.classes_

                # Map unseen values to the most frequent known value
                most_frequent = encoder.classes_[0]  # Assuming first class is most frequent
                data[col] = data[col].apply(lambda x: x if x in known_values else most_frequent)
                data[col] = encoder.transform(data[col])
            except Exception:
                # If encoding fails, fill with 0
                data[col] = 0

    # Ensure all required columns are present
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    # Reorder columns to match training data
    data = data[feature_columns]

    # Scale the features
    data_scaled = scaler.transform(data)

    # Make predictions
    predictions_proba = model.predict(data_scaled)
    predictions = (predictions_proba > 0.5).astype(int)

    # Create results dataframe
    results_df = original_data.copy()
    results_df['Prediction_Probability'] = predictions_proba.flatten()
    results_df['Prediction'] = ['Approved' if pred == 1 else 'Not Approved' for pred in predictions]
    results_df['Confidence'] = results_df['Prediction_Probability'].apply(
        lambda x: x if x > 0.5 else 1 - x
    )

    return results_df

def generate_groq_response(client, user_message, context=""):
    """Generate response using Groq API with configurable model"""
    try:
        # Get model from environment variable or use default
        model = os.getenv('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')

        system_prompt = f"""You are a helpful loan advisor assistant. You help users understand loan approval processes and gather necessary information for loan applications.

Context about the loan application system:
- The system predicts loan approval based on factors like income, credit history, education, employment status, etc.
- Required information: Gender, Marital Status, Dependents, Education, Employment Status, Applicant Income, Coapplicant Income, Loan Amount, Loan Term, Credit History, Property Area

Current context: {context}

Please provide helpful, accurate, and friendly responses about loan applications. If asked about specific loan details, guide the user through the application process step by step."""

        completion = client.chat.completions.create(
            model=model,
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

def extract_loan_details(text):
    """Extract loan application details from user text"""
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

# Sidebar
with st.sidebar:
    st.title("🏦 Loan Approval System")
    page = st.selectbox(
        "Select Page",
        ["Home", "Dataset Upload", "Data Quality", "Model Training", "Loan Prediction", "Bulk Prediction", "AI Chatbot", "Analytics"]
    )

# Main content based on selected page
if page == "Home":
    st.title("🏦 Loan Approval Prediction System")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        ### 📊 Dataset Upload
        Upload your loan dataset to train the prediction model.
        """)

    with col2:
        st.markdown("""
        ### 🤖 AI Model Training
        Train a deep learning model on your dataset for accurate predictions.
        """)

    with col3:
        st.markdown("""
        ### 📋 Bulk Prediction
        Upload test data for batch loan approval predictions.
        """)

    with col4:
        st.markdown("""
        ### 💬 AI Chatbot
        Interactive chatbot to help with loan applications and predictions.
        """)
    
    st.markdown("---")
    st.markdown("""
    ## How it works:
    1. **Upload Dataset**: Upload your loan dataset (CSV format)
    2. **Analyze Data Quality**: Review data quality metrics and visualizations
    3. **Train Model**: Train a deep learning model with overfitting prevention
    4. **Make Predictions**: Use the trained model to predict individual loan approvals
    5. **Bulk Predictions**: Upload test data for batch predictions with downloadable results
    6. **Chat with AI**: Get personalized assistance through our AI chatbot
    """)

elif page == "Dataset Upload":
    st.title("📊 Dataset Upload & Visualization")
    st.markdown("Upload your loan dataset to train the prediction model and explore comprehensive visualizations.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.dataset = df

            st.success("Dataset uploaded successfully!")

            # Dataset Overview
            st.header("📋 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                if 'Loan_Status' in df.columns:
                    approval_rate = (df['Loan_Status'] == 'Y').mean() * 100
                    st.metric("Approval Rate", f"{approval_rate:.1f}%")

            # Display sample data
            st.subheader("📄 Sample Data")
            st.dataframe(df.head(10))

            # Data Types and Missing Values
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Data Types")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values
                })
                st.dataframe(dtype_df)

            with col2:
                st.subheader("🔍 Missing Values Summary")
                missing_values = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing_values.index,
                    'Missing Count': missing_values.values,
                    'Missing Percentage': (missing_values.values / len(df)) * 100
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                if not missing_df.empty:
                    st.dataframe(missing_df)

                    # Missing values bar chart
                    fig = px.bar(missing_df, x='Column', y='Missing Percentage',
                                title='Missing Values by Column (%)')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No missing values found!")

            # Missing Values Heatmap
            if df.isnull().sum().sum() > 0:
                st.subheader("🔥 Missing Values Pattern")
                missing_heatmap = create_missing_values_heatmap(df)
                st.plotly_chart(missing_heatmap, use_container_width=True)

            # Data Distribution Analysis
            st.header("📈 Data Distribution Analysis")

            # Target variable distribution
            if 'Loan_Status' in df.columns:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🎯 Target Variable Distribution")
                    fig = px.pie(df, names='Loan_Status', title="Loan Approval Distribution")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("📊 Approval by Categories")
                    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
                    selected_cat = st.selectbox("Select categorical variable:", categorical_cols)

                    if selected_cat in df.columns:
                        fig = px.histogram(df, x=selected_cat, color='Loan_Status',
                                         title=f'Loan Status by {selected_cat}', barmode='group')
                        st.plotly_chart(fig, use_container_width=True)

            # Numerical Variables Distribution
            st.subheader("📊 Numerical Variables Distribution")
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Loan_ID' in numerical_cols:
                numerical_cols.remove('Loan_ID')

            if numerical_cols:
                selected_num_cols = st.multiselect("Select numerical columns to visualize:",
                                                 numerical_cols, default=numerical_cols[:3])

                if selected_num_cols:
                    # Create subplots for distributions
                    n_cols = min(2, len(selected_num_cols))
                    n_rows = (len(selected_num_cols) + n_cols - 1) // n_cols

                    for i, col in enumerate(selected_num_cols):
                        if i % 2 == 0:
                            col1, col2 = st.columns(2)

                        with col1 if i % 2 == 0 else col2:
                            fig = px.histogram(df, x=col, nbins=30, title=f'Distribution of {col}')
                            fig.add_vline(x=df[col].mean(), line_dash="dash", line_color="red",
                                        annotation_text=f"Mean: {df[col].mean():.2f}")
                            st.plotly_chart(fig, use_container_width=True)

            # Correlation Analysis
            st.subheader("🔗 Correlation Analysis")
            corr_heatmap = create_correlation_heatmap(df)
            if corr_heatmap:
                st.plotly_chart(corr_heatmap, use_container_width=True)
            else:
                st.info("Not enough numerical columns for correlation analysis.")

            # Outlier Detection
            st.subheader("🚨 Outlier Detection")
            if numerical_cols:
                selected_outlier_col = st.selectbox("Select column for outlier analysis:", numerical_cols)

                if selected_outlier_col:
                    col1, col2 = st.columns(2)

                    with col1:
                        # Box plot for outliers
                        fig = px.box(df, y=selected_outlier_col, title=f'Outliers in {selected_outlier_col}')
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Outlier statistics
                        outliers, lower_bound, upper_bound = detect_outliers_iqr(df, selected_outlier_col)

                        st.write("**Outlier Statistics:**")
                        st.write(f"Lower Bound: {lower_bound:.2f}")
                        st.write(f"Upper Bound: {upper_bound:.2f}")
                        st.write(f"Number of Outliers: {len(outliers)}")
                        st.write(f"Percentage of Outliers: {(len(outliers)/len(df))*100:.2f}%")

                        if len(outliers) > 0:
                            st.write("**Sample Outliers:**")
                            st.dataframe(outliers[[selected_outlier_col]].head())

            # Statistical Summary
            st.subheader("📊 Statistical Summary")
            if numerical_cols:
                st.dataframe(df[numerical_cols].describe())

            # Categorical Variables Analysis
            st.subheader("📋 Categorical Variables Analysis")
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if 'Loan_ID' in categorical_cols:
                categorical_cols.remove('Loan_ID')

            if categorical_cols:
                for col in categorical_cols:
                    with st.expander(f"Analysis of {col}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            # Value counts
                            value_counts = df[col].value_counts()
                            st.write("**Value Counts:**")
                            st.dataframe(value_counts.reset_index())

                        with col2:
                            # Bar chart
                            fig = px.bar(x=value_counts.index, y=value_counts.values,
                                       title=f'Distribution of {col}')
                            fig.update_layout(xaxis_title=col, yaxis_title='Count')
                            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading file: {e}")

elif page == "Data Quality":
    st.title("🔍 Data Quality Dashboard")

    if st.session_state.dataset is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state.dataset

        # Data Quality Overview
        st.header("📊 Data Quality Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")

        with col2:
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)

        with col3:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            total_outliers = 0
            for col in numerical_cols:
                if col != 'Loan_ID':
                    outliers, _, _ = detect_outliers_iqr(df, col)
                    total_outliers += len(outliers)
            st.metric("Total Outliers", total_outliers)

        with col4:
            categorical_cols = df.select_dtypes(include=['object']).columns
            inconsistencies = 0
            for col in categorical_cols:
                if col not in ['Loan_ID', 'Loan_Status']:
                    # Check for potential inconsistencies (case variations, etc.)
                    unique_vals = df[col].dropna().astype(str).str.strip().str.lower().nunique()
                    original_vals = df[col].dropna().nunique()
                    if unique_vals != original_vals:
                        inconsistencies += 1
            st.metric("Data Inconsistencies", inconsistencies)

        # Missing Values Analysis
        st.header("❌ Missing Values Analysis")

        missing_summary = df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

        if not missing_summary.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Missing Values by Column")
                missing_df = pd.DataFrame({
                    'Column': missing_summary.index,
                    'Missing Count': missing_summary.values,
                    'Missing Percentage': (missing_summary.values / len(df)) * 100
                })
                st.dataframe(missing_df)

            with col2:
                st.subheader("Missing Values Visualization")
                fig = px.bar(missing_df, x='Column', y='Missing Percentage',
                           title='Missing Values by Column (%)')
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            # Missing values pattern
            st.subheader("Missing Values Pattern")
            missing_heatmap = create_missing_values_heatmap(df)
            st.plotly_chart(missing_heatmap, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")

        # Outlier Analysis
        st.header("🚨 Outlier Analysis")

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Loan_ID' in numerical_cols:
            numerical_cols.remove('Loan_ID')

        if numerical_cols:
            outlier_summary = []

            for col in numerical_cols:
                outliers, lower_bound, upper_bound = detect_outliers_iqr(df, col)
                outlier_summary.append({
                    'Column': col,
                    'Outlier Count': len(outliers),
                    'Outlier Percentage': (len(outliers) / len(df)) * 100,
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound
                })

            outlier_df = pd.DataFrame(outlier_summary)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Outlier Summary")
                st.dataframe(outlier_df)

            with col2:
                st.subheader("Outlier Visualization")
                fig = px.bar(outlier_df, x='Column', y='Outlier Percentage',
                           title='Outliers by Column (%)')
                st.plotly_chart(fig, use_container_width=True)

            # Detailed outlier analysis for selected column
            st.subheader("Detailed Outlier Analysis")
            selected_col = st.selectbox("Select column for detailed analysis:", numerical_cols)

            if selected_col:
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.box(df, y=selected_col, title=f'Box Plot - {selected_col}')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.histogram(df, x=selected_col, nbins=30, title=f'Distribution - {selected_col}')
                    st.plotly_chart(fig, use_container_width=True)

        # Data Preprocessing Simulation
        st.header("🔧 Data Preprocessing Simulation")

        st.subheader("Before vs After Preprocessing")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Dataset:**")
            st.write(f"Shape: {df.shape}")
            st.write(f"Missing Values: {df.isnull().sum().sum()}")
            st.write(f"Data Types: {df.dtypes.value_counts().to_dict()}")

        with col2:
            # Simulate preprocessing
            try:
                X_processed, y_processed, _, feature_cols, prep_steps = preprocess_data(df)

                st.write("**After Preprocessing:**")
                st.write(f"Shape: {X_processed.shape}")
                st.write(f"Missing Values: {X_processed.isnull().sum().sum()}")
                st.write(f"Features: {len(feature_cols)}")

                st.subheader("Preprocessing Steps")
                for i, step in enumerate(prep_steps, 1):
                    st.write(f"{i}. {step}")

            except Exception as e:
                st.error(f"Error in preprocessing simulation: {e}")

elif page == "Model Training":
    st.title("🤖 Enhanced Model Training")

    if st.session_state.dataset is None:
        st.warning("Please upload a dataset first.")
    else:
        st.write("Dataset loaded successfully!")

        # Preprocessing Options
        st.subheader("🔧 Preprocessing Options")
        col1, col2 = st.columns(2)

        with col1:
            handle_outliers_flag = st.checkbox("Handle Outliers", value=True,
                                              help="Remove or cap outliers using IQR method")
            feature_engineering_flag = st.checkbox("Feature Engineering", value=True,
                                                   help="Create new features from existing ones")

        with col2:
            st.info("**Preprocessing Steps:**\n"
                   "1. Handle missing values\n"
                   "2. Feature engineering (if enabled)\n"
                   "3. Outlier treatment (if enabled)\n"
                   "4. Categorical encoding\n"
                   "5. Feature scaling")

        if st.button("Train Model", type="primary"):
            try:
                with st.spinner("Preprocessing data..."):
                    # Preprocess data with options
                    X, y, label_encoders, feature_columns, preprocessing_steps = preprocess_data(
                        st.session_state.dataset,
                        handle_outliers_flag=handle_outliers_flag,
                        feature_engineering_flag=feature_engineering_flag
                    )

                # Display preprocessing steps
                st.subheader("📋 Preprocessing Steps Applied")
                for i, step in enumerate(preprocessing_steps, 1):
                    st.write(f"{i}. {step}")

                # Show dataset shape after preprocessing
                st.info(f"Dataset shape after preprocessing: {X.shape}")

                with st.spinner("Training the model..."):
                    # Train model
                    (model, scaler, train_acc, test_acc, history, feature_importance,
                     feature_names, classification_rep, confusion_mat) = train_model(X, y)

                # Store in session state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.label_encoders = label_encoders
                st.session_state.feature_columns = feature_columns
                st.session_state.model_trained = True

                st.success("Model trained successfully!")

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Training Accuracy", f"{train_acc:.4f}")

                with col2:
                    st.metric("Test Accuracy", f"{test_acc:.4f}")

                with col3:
                    overfitting = train_acc - test_acc
                    st.metric("Overfitting", f"{overfitting:.4f}",
                             delta=f"{'Good' if overfitting < 0.05 else 'Check'}")

                # Training History
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📈 Training History")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Training Accuracy'))
                    fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy'))
                    fig.update_layout(title='Model Accuracy Over Time', xaxis_title='Epoch', yaxis_title='Accuracy')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("📉 Loss History")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                    fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                    fig.update_layout(title='Model Loss Over Time', xaxis_title='Epoch', yaxis_title='Loss')
                    st.plotly_chart(fig, use_container_width=True)

                # Feature Importance
                st.subheader("🎯 Feature Importance")
                feature_imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False)

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(feature_imp_df.head(10), x='Importance', y='Feature',
                               orientation='h', title='Top 10 Most Important Features')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.dataframe(feature_imp_df)

                # Model Performance Metrics
                st.subheader("📊 Detailed Performance Metrics")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Classification Report:**")
                    # Convert classification report to DataFrame for better display
                    class_df = pd.DataFrame(classification_rep).transpose()
                    st.dataframe(class_df.round(3))

                with col2:
                    st.write("**Confusion Matrix:**")
                    fig = px.imshow(confusion_mat,
                                   text_auto=True,
                                   aspect="auto",
                                   title="Confusion Matrix",
                                   labels=dict(x="Predicted", y="Actual"),
                                   x=['Not Approved', 'Approved'],
                                   y=['Not Approved', 'Approved'])
                    st.plotly_chart(fig, use_container_width=True)

                # Training insights
                st.subheader("🧠 Training Insights")

                # Check for overfitting
                final_train_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                overfitting_gap = final_train_acc - final_val_acc

                col1, col2, col3 = st.columns(3)

                with col1:
                    if overfitting_gap < 0.05:
                        st.success(f"✅ Good generalization (gap: {overfitting_gap:.3f})")
                    elif overfitting_gap < 0.1:
                        st.warning(f"⚠️ Slight overfitting (gap: {overfitting_gap:.3f})")
                    else:
                        st.error(f"❌ Overfitting detected (gap: {overfitting_gap:.3f})")

                with col2:
                    epochs_trained = len(history.history['accuracy'])
                    st.info(f"🔄 Epochs trained: {epochs_trained}")

                with col3:
                    best_val_acc = max(history.history['val_accuracy'])
                    st.info(f"🎯 Best validation accuracy: {best_val_acc:.4f}")

                # Recommendations
                st.subheader("💡 Recommendations")

                if overfitting_gap > 0.1:
                    st.warning("""
                    **High overfitting detected. Consider:**
                    - Increasing dropout rates
                    - Adding more regularization
                    - Reducing model complexity
                    - Getting more training data
                    """)
                elif overfitting_gap < 0.02:
                    st.success("""
                    **Excellent model generalization!**
                    - Model is well-balanced
                    - Good training/validation performance
                    - Ready for deployment
                    """)
                else:
                    st.info("""
                    **Good model performance.**
                    - Acceptable generalization
                    - Monitor performance on new data
                    """)

            except Exception as e:
                st.error(f"Error training model: {e}")
                st.exception(e)

elif page == "Loan Prediction":
    st.title("💳 Loan Prediction")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first.")
    else:
        st.write("Enter the applicant details for loan prediction:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
            
        with col2:
            applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
            loan_term = st.number_input("Loan Amount Term", min_value=0, value=360)
            credit_history = st.selectbox("Credit History", [1, 0])
        
        if st.button("Predict Loan Approval", type="primary"):
            user_input = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_term,
                'Credit_History': credit_history,
                'Property_Area': property_area
            }
            
            try:
                prediction, probability = predict_loan_approval(
                    st.session_state.model,
                    st.session_state.scaler,
                    st.session_state.label_encoders,
                    st.session_state.feature_columns,
                    user_input
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == "Approved":
                        st.success(f"🎉 Loan {prediction}!")
                    else:
                        st.error(f"❌ Loan {prediction}")
                        
                with col2:
                    st.metric("Confidence", f"{probability:.2%}")
                
                # Display probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Approval Probability"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")

elif page == "Bulk Prediction":
    st.title("📊 Bulk Loan Prediction")

    if not st.session_state.model_trained:
        st.warning("Please train a model first before making bulk predictions.")
    else:
        st.write("Upload a CSV file with loan application data to get predictions for multiple applicants at once.")

        # File upload
        uploaded_test_file = st.file_uploader("Choose a test CSV file", type="csv", key="bulk_prediction")

        if uploaded_test_file is not None:
            try:
                # Read the test file
                test_df = pd.read_csv(uploaded_test_file)

                st.success(f"Test file uploaded successfully! Found {len(test_df)} records.")

                # Display sample of uploaded data
                st.subheader("📄 Sample Test Data")
                st.dataframe(test_df.head())

                # Show data info
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Records", len(test_df))

                with col2:
                    st.metric("Total Columns", len(test_df.columns))

                with col3:
                    missing_values = test_df.isnull().sum().sum()
                    st.metric("Missing Values", missing_values)

                # Check for required columns
                required_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                                  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                  'Loan_Amount_Term', 'Credit_History', 'Property_Area']

                missing_columns = [col for col in required_columns if col not in test_df.columns]

                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                    st.info("Please ensure your test file contains all required columns for prediction.")
                else:
                    st.success("✅ All required columns found!")

                    # Prediction options
                    st.subheader("🔧 Prediction Options")

                    col1, col2 = st.columns(2)

                    with col1:
                        confidence_threshold = st.slider(
                            "Confidence Threshold",
                            min_value=0.5,
                            max_value=0.95,
                            value=0.7,
                            step=0.05,
                            help="Minimum confidence level for approval recommendations"
                        )

                    with col2:
                        show_details = st.checkbox("Show detailed analysis", value=True)

                    # Make predictions button
                    if st.button("🚀 Generate Bulk Predictions", type="primary"):
                        with st.spinner("Making predictions for all records..."):
                            try:
                                # Make bulk predictions
                                results_df = predict_bulk_loan_approval(
                                    st.session_state.model,
                                    st.session_state.scaler,
                                    st.session_state.label_encoders,
                                    st.session_state.feature_columns,
                                    test_df
                                )

                                st.success("✅ Predictions completed successfully!")

                                # Summary statistics
                                st.subheader("📊 Prediction Summary")

                                col1, col2, col3, col4 = st.columns(4)

                                total_records = len(results_df)
                                approved_count = len(results_df[results_df['Prediction'] == 'Approved'])
                                high_confidence = len(results_df[results_df['Confidence'] >= confidence_threshold])
                                avg_confidence = results_df['Confidence'].mean()

                                with col1:
                                    st.metric("Total Applications", total_records)

                                with col2:
                                    st.metric("Approved", approved_count)
                                    st.caption(f"{(approved_count/total_records)*100:.1f}% approval rate")

                                with col3:
                                    st.metric("High Confidence", high_confidence)
                                    st.caption(f"≥{confidence_threshold:.0%} confidence")

                                with col4:
                                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")

                                # Visualizations
                                if show_details:
                                    st.subheader("📈 Prediction Analysis")

                                    col1, col2 = st.columns(2)

                                    with col1:
                                        # Prediction distribution
                                        fig = px.pie(results_df, names='Prediction',
                                                   title="Loan Approval Distribution")
                                        st.plotly_chart(fig, use_container_width=True)

                                    with col2:
                                        # Confidence distribution
                                        fig = px.histogram(results_df, x='Confidence', nbins=20,
                                                         title="Confidence Score Distribution")
                                        fig.add_vline(x=confidence_threshold, line_dash="dash",
                                                     line_color="red", annotation_text="Threshold")
                                        st.plotly_chart(fig, use_container_width=True)

                                # Results table
                                st.subheader("📋 Detailed Results")

                                # Filter options
                                col1, col2 = st.columns(2)

                                with col1:
                                    filter_prediction = st.selectbox(
                                        "Filter by Prediction",
                                        ["All", "Approved", "Not Approved"]
                                    )

                                with col2:
                                    filter_confidence = st.selectbox(
                                        "Filter by Confidence",
                                        ["All", f"High (≥{confidence_threshold:.0%})", f"Low (<{confidence_threshold:.0%})"]
                                    )

                                # Apply filters
                                filtered_df = results_df.copy()

                                if filter_prediction != "All":
                                    filtered_df = filtered_df[filtered_df['Prediction'] == filter_prediction]

                                if filter_confidence == f"High (≥{confidence_threshold:.0%})":
                                    filtered_df = filtered_df[filtered_df['Confidence'] >= confidence_threshold]
                                elif filter_confidence == f"Low (<{confidence_threshold:.0%})":
                                    filtered_df = filtered_df[filtered_df['Confidence'] < confidence_threshold]

                                # Display filtered results
                                st.dataframe(filtered_df, use_container_width=True)

                                # Download button
                                st.subheader("💾 Download Results")

                                # Prepare CSV for download
                                csv_buffer = io.StringIO()
                                results_df.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()

                                st.download_button(
                                    label="📥 Download Predictions as CSV",
                                    data=csv_data,
                                    file_name=f"loan_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    type="primary"
                                )

                                # Additional insights
                                if show_details:
                                    st.subheader("🔍 Additional Insights")

                                    # Risk analysis
                                    high_risk = results_df[
                                        (results_df['Prediction'] == 'Approved') &
                                        (results_df['Confidence'] < confidence_threshold)
                                    ]

                                    if len(high_risk) > 0:
                                        st.warning(f"⚠️ {len(high_risk)} approved applications have low confidence. Consider manual review.")
                                        with st.expander("View Low Confidence Approvals"):
                                            st.dataframe(high_risk)

                                    # High confidence rejections
                                    confident_rejections = results_df[
                                        (results_df['Prediction'] == 'Not Approved') &
                                        (results_df['Confidence'] >= 0.8)
                                    ]

                                    if len(confident_rejections) > 0:
                                        st.info(f"ℹ️ {len(confident_rejections)} applications were rejected with high confidence.")

                            except Exception as e:
                                st.error(f"Error making bulk predictions: {e}")
                                st.exception(e)

            except Exception as e:
                st.error(f"Error reading test file: {e}")

        else:
            # Show expected file format
            st.subheader("📋 Expected File Format")
            st.write("Your test CSV file should contain the following columns:")

            expected_columns = [
                "Gender", "Married", "Dependents", "Education", "Self_Employed",
                "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "Credit_History", "Property_Area"
            ]

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Required Columns:**")
                for col in expected_columns[:6]:
                    st.write(f"• {col}")

            with col2:
                st.write("**Additional Columns:**")
                for col in expected_columns[6:]:
                    st.write(f"• {col}")

            # Sample data format
            st.subheader("📄 Sample Data Format")
            sample_data = {
                'Gender': ['Male', 'Female', 'Male'],
                'Married': ['Yes', 'No', 'Yes'],
                'Dependents': ['0', '1', '2'],
                'Education': ['Graduate', 'Graduate', 'Not Graduate'],
                'Self_Employed': ['No', 'Yes', 'No'],
                'ApplicantIncome': [5849, 4583, 3000],
                'CoapplicantIncome': [0, 1508, 0],
                'LoanAmount': [128, 128, 66],
                'Loan_Amount_Term': [360, 360, 360],
                'Credit_History': [1, 1, 1],
                'Property_Area': ['Urban', 'Rural', 'Urban']
            }

            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df)

elif page == "AI Chatbot":
    st.title("💬 AI Loan Assistant")
    
    # Initialize Groq client
    client = initialize_groq_client()
    
    if client is None:
        st.error("Failed to initialize AI assistant. Please check your API key.")
    else:
        st.write("Chat with our AI assistant to get help with your loan application!")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if user_input := st.chat_input("Ask me about loan applications..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.chat_message("user"):
                st.write(user_input)
            
            # Extract loan details from user input
            extracted_details = extract_loan_details(user_input)
            st.session_state.user_data.update(extracted_details)
            
            # Generate AI response
            context = f"User data collected so far: {st.session_state.user_data}"
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_groq_response(client, user_input, context)
                    st.write(response)
                    
                    # If we have enough data and model is trained, offer prediction
                    if (st.session_state.model_trained and 
                        len(st.session_state.user_data) >= 5 and
                        'ApplicantIncome' in st.session_state.user_data):
                        
                        if st.button("Get Loan Prediction", key="chat_predict"):
                            # Fill missing values with defaults
                            default_values = {
                                'Gender': 'Male',
                                'Married': 'No',
                                'Dependents': '0',
                                'Education': 'Graduate',
                                'Self_Employed': 'No',
                                'ApplicantIncome': 5000,
                                'CoapplicantIncome': 0,
                                'LoanAmount': 100,
                                'Loan_Amount_Term': 360,
                                'Credit_History': 1,
                                'Property_Area': 'Urban'
                            }
                            
                            user_input_complete = {**default_values, **st.session_state.user_data}
                            
                            try:
                                prediction, probability = predict_loan_approval(
                                    st.session_state.model,
                                    st.session_state.scaler,
                                    st.session_state.label_encoders,
                                    st.session_state.feature_columns,
                                    user_input_complete
                                )
                                
                                pred_response = f"Based on your information, your loan is likely to be **{prediction}** with a confidence of {probability:.2%}."
                                st.write(pred_response)
                                
                            except Exception as e:
                                st.error(f"Error making prediction: {e}")
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.user_data = {}
            st.rerun()

elif page == "Analytics":
    st.title("📈 Analytics Dashboard")
    
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state.dataset
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Applications", len(df))
        with col2:
            approved = (df['Loan_Status'] == 'Y').sum()
            st.metric("Approved Loans", approved)
        with col3:
            approval_rate = (approved / len(df)) * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        with col4:
            avg_loan_amount = df['LoanAmount'].mean()
            st.metric("Avg Loan Amount", f"${avg_loan_amount:.0f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            fig = px.histogram(df, x='Gender', color='Loan_Status', 
                             title='Loan Status by Gender')
            st.plotly_chart(fig)
            
        with col2:
            # Education distribution
            fig = px.histogram(df, x='Education', color='Loan_Status', 
                             title='Loan Status by Education')
            st.plotly_chart(fig)
        
        # Income analysis
        st.subheader("Income Analysis")
        fig = px.box(df, x='Loan_Status', y='ApplicantIncome', 
                    title='Applicant Income by Loan Status')
        st.plotly_chart(fig)
        
        # Credit history impact
        st.subheader("Credit History Impact")
        credit_analysis = df.groupby(['Credit_History', 'Loan_Status']).size().unstack()
        fig = px.bar(credit_analysis, title='Loan Status by Credit History')
        st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Groq AI")