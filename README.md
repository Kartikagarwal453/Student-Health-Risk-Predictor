# 🏥 AI-Powered Student Health Risk Prediction System

An end-to-end Machine Learning application that predicts a student's health risk using lifestyle, physiological, and behavioral indicators. The project demonstrates the complete Data Science lifecycle, from data preprocessing and feature engineering to model deployment through an interactive Streamlit dashboard.

---

## 🚀 Live Demo

🌐 **Application:** https://student-health-risk-predictor-by-kartik.streamlit.app/

💻 **GitHub Repository:** https://github.com/Kartikagarwal453/Student-Health-Risk-Predictor

---

# 📌 Project Overview

The **AI-Powered Student Health Risk Prediction System** predicts whether a student belongs to one of the following health categories:

- ✅ Fit
- ⚠️ At Risk
- ❌ Unhealthy

The prediction is based on multiple lifestyle, health, and personal attributes such as sleep duration, BMI, heart rate, exercise duration, physical activity, stress level, diet type, water intake, and more.

> **Note:** This project was built using a publicly available Kaggle dataset. The complete machine learning pipeline, feature engineering, model comparison, optimization, interactive dashboard, and deployment were independently designed and implemented.

---

# ✨ Features

- 🩺 Individual Student Health Prediction
- 📂 Batch Prediction using CSV Upload
- 📊 Interactive Dashboard
- 📈 Dataset Insights
- 🤖 Model Insights
- 📋 Personalized Health Recommendations
- 📄 Downloadable PDF Health Report
- 🌙 Modern Responsive User Interface
- ☁️ Streamlit Cloud Deployment

---

# 📊 Data Science Workflow

## 1️⃣ Data Cleaning

- Removed duplicate records
- Handled missing values
- Corrected data types
- Prepared data for machine learning

---

## 2️⃣ Exploratory Data Analysis (EDA)

Performed detailed analysis including:

- Class Distribution
- Feature Distribution
- Correlation Analysis
- Health Trend Analysis
- Lifestyle Pattern Analysis
- Outlier Detection

---

## 3️⃣ Feature Engineering

Created meaningful features to improve prediction performance:

- Lifestyle Score
- Sleep Categories
- BMI Categories
- Physical Activity Features
- Interaction Features

---

## 4️⃣ Machine Learning Models

The following classification algorithms were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees Classifier
- AdaBoost
- Gradient Boosting
- HistGradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Artificial Neural Network (TensorFlow)

---

# 🏆 Best Performing Model

**XGBoost Classifier**

### Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | **96.59%** |
| Balanced Accuracy | **90%+** |
| Macro F1 Score | **90.48%** |

---

# 📊 Dashboard Modules

### 🏠 Home

- Project Overview
- Model Performance
- Quick Statistics

### 🩺 Individual Prediction

Predict the health risk of a single student using health and lifestyle information.

### 📂 Batch Prediction

Upload a CSV file and generate predictions for multiple students.

### 📈 Dataset Insights

Interactive visualizations and exploratory data analysis.

### 🤖 Model Insights

- Model Comparison
- Feature Importance
- Performance Metrics
- Evaluation Results

### ℹ️ About

Project architecture, technologies used, and development workflow.

---

# 🛠️ Technology Stack

## Programming

- Python

## Data Analysis

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- XGBoost
- TensorFlow / Keras

## Data Visualization

- Plotly
- Matplotlib

## Web Application

- Streamlit

## Deployment

- Streamlit Community Cloud

## Version Control

- Git
- GitHub

---

# 📁 Project Structure

```text
Student-Health-Risk-Predictor/
│
├── assets/
├── dataset/
├── model/
├── pages/
├── utils/
│
├── app.py
├── style.css
├── requirements.txt
├── README.md
└── .streamlit/
```

---

# ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Kartikagarwal453/Student-Health-Risk-Predictor.git
```

### Navigate to the project folder

```bash
cd Student-Health-Risk-Predictor
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```


# 📚 Dataset

**Dataset Source:** Public **Student Health Risk Prediction** dataset from a Kaggle competition.

The dataset served as the foundation for building this end-to-end machine learning application.

## My Contribution

I independently designed and implemented the complete solution, including:

- ✅ Data Cleaning and Preprocessing
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature Engineering and Feature Selection
- ✅ Comparison of 10+ Machine Learning Models
- ✅ Hyperparameter Tuning and Cross-Validation
- ✅ XGBoost Model Optimization
- ✅ Artificial Neural Network (TensorFlow) Implementation for Comparison
- ✅ Model Evaluation using Accuracy, Balanced Accuracy, and Macro F1 Score
- ✅ Interactive Streamlit Dashboard Development
- ✅ Batch Prediction Functionality
- ✅ Personalized Health Recommendations
- ✅ PDF Report Generation
- ✅ Streamlit Cloud Deployment

> **Note:** While the dataset was sourced from a public Kaggle competition, all preprocessing, analysis, feature engineering, machine learning model development, evaluation, dashboard implementation, UI design, and deployment were completed independently.

---

# 🔮 Future Improvements

- SHAP Explainability
- REST API Integration
- Docker Deployment
- User Authentication
- Database Integration
- Prediction History
- CI/CD Pipeline
- Continuous Model Monitoring

---

# 🙏 Acknowledgements

This project uses the **Student Health Risk Prediction** dataset made publicly available through Kaggle.

Special thanks to the dataset contributors and the Kaggle community for providing the dataset used as the foundation for this project.

All data preprocessing, exploratory data analysis, feature engineering, machine learning model development, evaluation, dashboard implementation, UI design, and deployment were independently completed by me.

---

# 👨‍💻 Author

**Kartik Agarwal**

🎓 B.Tech Computer Science & Engineering (AI & ML)

Meerut Institute of Engineering & Technology (MIET)

🔗 **GitHub:** https://github.com/Kartikagarwal453

💼 **LinkedIn:** https://www.linkedin.com/in/kartik-agarwall/

🌐 **Live Demo:** https://student-health-risk-predictor-by-kartik.streamlit.app/

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It motivates me to continue building and sharing more Machine Learning and Data Science projects!
