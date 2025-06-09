# 🚕 Rideshare Price Prediction (Uber & Lyft)  
*Boston, Massachusetts - Machine Learning Project*

![Uber & Lyft](https://img.shields.io/badge/Machine%20Learning-Regression-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-green) ![Models](https://img.shields.io/badge/Best%20Model-CatBoost-orange)

## 📌 Project Overview

This project tackles the challenge of **accurately predicting rideshare prices** for Uber and Lyft services in **Boston, MA**. Using a machine learning pipeline, I analyzed historical ride data to develop a predictive model that allows users to **estimate ride prices based on ride configurations**.

To make the solution user-friendly, I also developed an **interactive web UI using Streamlit**, where users can simulate ride fare estimates.

**Business Value:**  
- Helps **riders** plan and optimize their rides  
- Offers **rideshare companies** insights into key price-driving factors  
- Supports **marketing & pricing strategy** planning  

---

## 🎯 Objectives

✅ Identify key factors influencing Uber & Lyft ride pricing  
✅ Build and compare regression models for fare prediction  
✅ Determine the best performing model  
✅ Develop an interactive price prediction interface  
✅ Provide insights into pricing patterns & business implications  

---

## 📂 Dataset

- Source: [Kaggle - Uber and Lyft Cab Prices](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices)  
- Location: Boston, MA  
- Size: ~693k records → ~625k after preprocessing  

### Features
- `distance`, `cab_type`, `source`, `destination`, `time_stamp`, `price`, `surge_multiplier`, `name` (service type)
- Initially attempted to integrate **weather.csv** — see Experimentation section.

---

## ⚙️ Methodology

### Data Preprocessing
- Removed missing values (~8%)  
- No duplicates present  
- Timestamp conversion → extracted `hour`, `day`, `weekday`, `is_peak_hour`  
- One-hot encoding for categorical features  
- Feature scaling with Min-Max Scaler (`distance`, `surge_multiplier`)  
- Outlier removal using IQR method  

### Machine Learning Models Used
| Model               | RMSE   | R² Score |
|---------------------|--------|----------|
| Linear Regression   | 2.1759 | 0.9378   |
| Random Forest       | 1.6958 | 0.9622   |
| XGBoost             | 1.6229 | 0.9654   |
| **CatBoost (Tuned)**| **1.5286** | **0.9693** |

### Feature Importance (Top Features)
- `name_Lux Black XL`
- `name_Black SUV`
- `distance`
- `name_Shared`, `name_UberPool`
- `cab_type_Uber`

*"This project helped me understand the importance of building a complete ML pipeline — from normalization and feature engineering to model selection, tuning, and evaluation, ensuring each step contributes meaningfully to the final outcome."*

---

## 🖥️ User Interface (Streamlit)

An interactive UI allows users to:
- Select **cab type**  
- Choose **source** and **destination**  
- Input **hour of the day**  
- Obtain **fare prediction instantly**

The interface automatically scales distance and applies surge multiplier based on historical patterns.

---

## 🛠️ Challenges Faced & Lessons Learned

### 🌀 1. Weather Data Integration Attempt
- Kaggle provided an additional `weather.csv` dataset.
- I attempted to merge `cab_rides.csv` + `weather.csv`.
- Result: **All models produced unrealistically low RMSE (~0.01)** → likely due to:
  - **Data leakage**
  - Poor feature engineering on timestamp alignment.
- Outcome: Discarded weather features after confirming underfitting in feature importance.

### 🧠 2. Neural Networks & SVM Exploration
- Explored **Neural Networks (Keras/TensorFlow)** and **SVM (SVR)** for complex non-linear relationships.
- Could not complete training on **local 4GB RAM machine** → memory errors.
- Lesson: **Cloud computing is essential** for scaling ML on large datasets.

### ⚙️ 3. Hyperparameter Tuning Challenges
- Initially tried **GridSearchCV** → too computationally expensive locally.
- Switched to **RandomizedSearchCV** → provided efficient and effective tuning.

### ☁️ 4. Learning: Need for Cloud Resources
- Attempted larger experiments taught me that **local machine capacity is a serious limitation**.
- Motivated me to explore **GCP / AWS** for future large-scale ML work.

---

## 💡 Business Value & Impact

### For Riders
- Enables better **ride planning** and **cost estimation**.
- Helps optimize choice of ride type and timing.

### For Rideshare Companies
- Understand which **factors drive pricing** the most.
- Support decisions on **dynamic pricing** and **premium services**.
- Opportunity to create **personalized offers** based on demand patterns.

### For Data Science & Analytics
- Demonstrates that **feature engineering & proper preprocessing** greatly impact performance.
- Shows value of using **advanced models like CatBoost** for mixed categorical + numerical data.

---

## 🗂 Project Structure

├── app.py # Streamlit app
├── catboost_fare_model.pkl # Trained CatBoost model
├── ML_Taxi_Fare_Prediction (3) (1).ipynb # Jupyter Notebook with full training pipeline
├── requirements.txt # List of required Python packages
├── scaler.joblib # Saved MinMaxScaler (temporary — to be removed later)
├── README.md # Project documentation (this file)
└── Ruthveek_ML_Rideshare_Price_Prediction_Report.pdf # Detailed project report

---

🚀 Live App

👉 [Click here to try the Rideshare Price Prediction App!](https://rideshare-price-prediction-afhvgrse6snhgbusecldfs.streamlit.app/)

---

🚀 Future Work

- Revisit **weather feature integration** with better time alignment and feature engineering.
- Add **real-time traffic and weather APIs** to enhance predictions.
- Extend model to **multiple cities** to test generalization.
- Explore **stacked ensemble models** for further accuracy gains.
- Optimize app performance by pre-saving distance_df, surge_df, and unique lists → faster app.

---

🧑‍💻 Author

**Ruthveek M R**  
Department of Data Science & Computer Applications  
MIT Manipal (2027 Batch)  
Email: [ruthmys123@gmail.com](mailto:ruthmys123@gmail.com)

---

🤝 Acknowledgements

- Dataset Source: [Kaggle - Uber and Lyft Cab Prices](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices)  
- Libraries: Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, Streamlit, Seaborn, Matplotlib
*"This project also helped me explore practical trade-offs between working within local hardware limitations and leveraging cloud-based platforms to scale and deploy machine learning workflows."*

---

*"Through this project, I gained hands-on experience in building a complete ML pipeline — from perfect data preprocessing to model tuning and deployment, while learning to overcome practical limitations along the way."* 🚀


