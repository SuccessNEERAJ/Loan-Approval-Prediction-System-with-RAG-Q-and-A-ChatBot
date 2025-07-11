# 🏦 Loan Prediction & AI Chatbot System

A comprehensive machine learning system for loan approval prediction with AI chatbot integration, built with Streamlit and TensorFlow.

## 🔗 Website Link - https://loan-approval-prediction-system-with-rag-q-and-a-chatbot.streamlit.app/

## ✨ Features

- **Advanced Data Preprocessing**: Missing value handling, outlier detection, and feature engineering
- **Deep Learning Model**: Neural network with regularization and early stopping to prevent overfitting
- **Data Quality Dashboard**: Comprehensive data analysis and visualization
- **Individual Predictions**: Single loan application predictions
- **Bulk Predictions**: Batch processing with downloadable results
- **AI Chatbot**: Interactive loan advisor powered by Groq AI
- **Comprehensive Analytics**: Model performance metrics and visualizations

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SuccessNEERAJ/Loan-Approval-Prediction-System-with-RAG-Q-and-A-ChatBot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your Groq API key:
   ```bash
   # Get your API key from: https://console.groq.com/
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

### 4. Run the Application

```bash
streamlit run Streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 🔑 API Key Setup

### Getting a Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and add it to your `.env` file

### Environment Variables

The following environment variables can be configured in your `.env` file:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GROQ_API_KEY` | Your Groq AI API key | Yes | None |
| `GROQ_MODEL` | Groq model to use | No | `meta-llama/llama-4-scout-17b-16e-instruct` |
| `STREAMLIT_SERVER_PORT` | Streamlit server port | No | `8501` |
| `DEFAULT_EPOCHS` | Default training epochs | No | `200` |
| `DEFAULT_BATCH_SIZE` | Default batch size | No | `16` |

## 📁 Project Structure

```
loan-prediction-chatbot/
├── Streamlit_app.py              # Main Streamlit application
├── Loan_Prediction_&_AI_Chatbot.py  # Core ML system implementation
├── Dataset/
│   └── Training/
│       └── Training Dataset.csv   # Training dataset
├── .env.example                   # Example environment variables
├── .env                          # Your actual environment variables (not in git)
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 📸 Application Screenshots

### 1. Home Page
![Home Page](Screenshots/1.%20Home%20Page.png)
*Welcome to the Loan Prediction & AI Chatbot System*

### 2. Dataset Upload & Visualization
![Dataset Upload](Screenshots/2.%20Dataset%20Upload.png)
*Upload your loan dataset and explore comprehensive visualizations*

![Dataset Overview](Screenshots/3.%20Dataset%20Uploaded.png)
*View dataset overview with key metrics and sample data*

### 3. Data Quality Dashboard
![Data Quality Overview](Screenshots/4.%20Data%20Quality%20Missing%20Values.png)
*Comprehensive data quality analysis with missing values visualization*

![Outlier Analysis](Screenshots/4.%20Data%20Quality%20Outliers.png)
*Detailed outlier detection and analysis*

### 4. Model Training
![Model Training Options](Screenshots/5.%20Model%20Training%20Page.png)
*Configure preprocessing options and train the model*

![Training Results](Screenshots/6.%20Model%20Accuracy%20and%20loss.png)
*View training accuracy, loss history, and overfitting analysis*

![Feature Importance](Screenshots/7.%20Feature%20Importance.png)
*Analyze feature importance and model interpretability*

![Performance Metrics](Screenshots/8.%20Model%20Performance%20Metrics.png)
*Detailed performance metrics with confusion matrix*

### 5. Individual Loan Prediction
![Loan Prediction Form](Screenshots/9.%20Single%20Loan%20Prediction%20Page.png)
*Enter applicant details for individual loan prediction*

![Prediction Results](Screenshots/9.%20Single%20Loan%20Prediction%20Result.png)
*Get instant approval prediction with confidence gauge*

### 6. Bulk Prediction System
![Bulk Prediction Upload](Screenshots/10.%20Test%20Data%20Bulk%20Loan%20Prediction%20Upload.png)
*Upload test CSV file for batch predictions*

![Bulk Results](Screenshots/11.%20Bulk%20Loan%20Prediction.png)
*View detailed bulk prediction results with filtering options*

![Detailed Analysis](Screenshots/12.%20Detailed%20Bulk%20Loan%20Predictions%20Results.png)
*Additional insights and analysis for bulk predictions*

![Download Results](Screenshots/13.%20Download%20Bulk%20Loan%20Predictions%20as%20CSV.png)
*Download prediction results as CSV file*

### 6. AI Chatbot Assistant
![AI Chatbot](Screenshots/14.%20AI%20Loan%20Assistant%20ChatBot%20Using%20Groq%20API%20Key.png)
*Interactive AI loan advisor powered by Groq*

![Chatbot Conversation](Screenshots/15.%20Chatbot%20Followup%20Questions%20and%20Answers.png)
*Natural language conversation with loan guidance*

### 7. Analytics Dashboard
![Analytics Dashboard](Screenshots/16.%20Final%20Analystics%20Dashboard.png)
*Comprehensive analytics with loan approval insights*

![Income Analysis](Screenshots/17.%20Income%20Analysis.png)
*Detailed income analysis and distribution charts*

![Credit History Impact](Screenshots/18.%20Credit%20History%20Impact.png)
*Credit history impact on loan approval*

## 🎯 How to Use

### 1. Dataset Upload
- Upload your loan dataset (CSV format)
- View comprehensive data analysis and visualizations
- Check data quality metrics

### 2. Data Quality Dashboard
- Analyze missing values and outliers
- View correlation heatmaps
- Understand data distributions

### 3. Model Training
- Configure preprocessing options
- Train deep learning model with overfitting prevention
- View training metrics and feature importance

### 4. Individual Predictions
- Input loan application details
- Get instant approval predictions with confidence scores

### 5. Bulk Predictions
- Upload test CSV file
- Get batch predictions for multiple applications
- Download results with confidence metrics

### 6. AI Chatbot
- Interactive loan advisor
- Get guidance on loan applications
- Extract information from natural language

## 📊 Dataset Format

Your dataset should include the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `Gender` | Categorical | Male/Female |
| `Married` | Categorical | Yes/No |
| `Dependents` | Categorical | 0/1/2/3+ |
| `Education` | Categorical | Graduate/Not Graduate |
| `Self_Employed` | Categorical | Yes/No |
| `ApplicantIncome` | Numerical | Applicant's income |
| `CoapplicantIncome` | Numerical | Coapplicant's income |
| `LoanAmount` | Numerical | Loan amount (in thousands) |
| `Loan_Amount_Term` | Numerical | Loan term (in months) |
| `Credit_History` | Categorical | 0/1 (0=No, 1=Yes) |
| `Property_Area` | Categorical | Urban/Rural/Semiurban |
| `Loan_Status` | Categorical | Y/N (Target variable) |

## 🔒 Security

- **API Keys**: Never commit API keys to version control
- **Environment Variables**: Use `.env` file for sensitive configuration
- **Git Ignore**: `.env` file is automatically ignored by git
- **Example File**: Use `.env.example` as a template

## 🛠️ Development

### Running the Core System

```bash
python Loan_Prediction_&_AI_Chatbot.py
```

### Installing Development Dependencies

```bash
pip install -r requirements.txt
# Add development tools if needed
pip install pytest black flake8
```

## 📈 Model Performance

The system includes several features to ensure good model performance:

- **Regularization**: L1/L2 regularization to prevent overfitting
- **Early Stopping**: Automatic training termination when validation loss stops improving
- **Batch Normalization**: Improved training stability
- **Dropout**: Additional regularization technique
- **Learning Rate Scheduling**: Adaptive learning rate reduction

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues:

1. Check that your `.env` file is properly configured
2. Ensure all dependencies are installed
3. Verify your Groq API key is valid
4. Check the console for error messages

For additional support, please open an issue in the repository.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- AI powered by [Groq](https://groq.com/)
