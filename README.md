# Student Performance ML Pipeline

## Author:Kareeb Sadab, Aspiring Data Scientist 

## Project Scope & Sector
This project belongs to the **Educational Data Science** and **EdTech Analytics** sector. Its purpose is to **predict student academic performance** using a structured, automated machine learning pipeline. By leveraging demographic and academic features, the system identifies patterns and predicts outcomes, enabling educational institutions, e-learning platforms, and policymakers to:  

- Identify at-risk students early.  
- Personalize learning interventions.  
- Optimize curriculum planning.  
- Make data-driven academic decisions.  

---

## Overview
The **Student Performance ML Pipeline** is a robust machine learning system that automates the workflow from raw data ingestion to model deployment. The pipeline incorporates:  

- **Data Ingestion** – loading raw student datasets and splitting them into training and test sets.  
- **Data Preprocessing & Transformation** – cleaning data, encoding categorical features, scaling numeric values, and preparing it for model training.  
- **Model Training & Hyperparameter Tuning** – training multiple regression models including Random Forest, Decision Tree, Gradient Boosting, Linear Regression, K-Neighbors, XGBoost, CatBoost, and AdaBoost. Hyperparameters are optimized using **GridSearchCV**.  
- **Model Evaluation & Selection** – evaluating models using **R² score** and selecting the best-performing model automatically.  
- **Model Persistence** – saving the trained model for future inference or deployment.  

This approach ensures **reproducibility**, **scalability**, and **efficient performance evaluation**, making it a reliable tool for educational analytics.  

---

## How the Model Works
1. **Data Ingestion**: The system reads student data (e.g., grades, attendance, demographic info) and splits it into train and test sets. Raw, train, and test datasets are saved in an artifacts folder for reproducibility.  

2. **Data Transformation**: 
   - Handle missing values and outliers.  
   - Encode categorical variables (e.g., gender, school type).  
   - Scale numerical features to standardize the dataset.  
   - Output is a preprocessed NumPy array ready for model training.  

3. **Model Training & Hyperparameter Tuning**:  
   - Multiple regression models are instantiated.  
   - Hyperparameters for each model are tuned via **GridSearchCV** to find the best combination.  
   - Each model is trained on the training set and evaluated on the test set using the **R² score**.  

4. **Model Evaluation & Selection**:  
   - The system compares R² scores for all models.  
   - The model with the highest score (above a threshold, e.g., 0.6) is automatically selected as the best predictor.  

5. **Model Saving & Deployment**:  
   - The best model is serialized and saved (`artifacts/model.pkl`) for future inference.  
   - This allows deployment in production or integration with web/desktop applications.  

---

## Features
- Automated **data ingestion, preprocessing, and transformation**.  
- Support for multiple regression models: **Random Forest, Decision Tree, Gradient Boosting, Linear Regression, K-Neighbors, XGBoost, CatBoost, AdaBoost**.  
- **Hyperparameter tuning** for all models using GridSearchCV.  
- **Automatic model evaluation** and best model selection.  
- **R² score reporting** for model performance.  
- **Model persistence** for deployment-ready output.  
- Fully **modular and reproducible pipeline** for scalable ML workflows.  

---

## Installation


# Clone the repository
git clone https://github.com/<your-username>/student-performance-ml-pipeline.git
cd student-performance-ml-pipeline

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt





## Usage 

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Step 1: Data Ingestion
ingestion = DataIngestion()
train_path, test_path = ingestion.initiate_data_ingestion()

# Step 2: Data Transformation
transformation = DataTransformation()
train_array, test_array, preprocessor = transformation.initiate_data_transformation(train_path, test_path)

# Step 3: Model Training & Evaluation
trainer = ModelTrainer()
r2_score_value = trainer.initiate_model_trainer(train_array, test_array)
print(f"Best model R² score: {r2_score_value}")

## License

This project is licensed under the MIT License.

