AQI – Air Quality Intelligence System

An end-to-end Machine Learning powered Air Quality Intelligence System that analyzes historical AQI data, predicts high-risk pollution levels, and provides smart health advisories.

Built using Python, Pandas, Scikit-Learn, and Matplotlib.

🚀 Project Overview

Air pollution is one of the biggest health threats in urban India.
This system:
📊 Analyzes historical AQI data (2015–2025)
🤖 Predicts High Risk pollution days using ML
📈 Visualizes AQI trends over time
⚠️ Generates next-day risk forecasts
🏥 Provides smart health advisories

🧠 Machine Learning Approach
🔹 High-Risk Classification Model

Target: High_Risk (AQI > 200)
Algorithm: Random Forest Classifier
Train-Test Split: 80-20
City Used: Delhi
Total Records: 18,265

📊 Model Performance
Accuracy: 0.556

Class 0 (Normal):
Precision: 0.41
Recall: 0.25

Class 1 (High Risk):
Precision: 0.60
Recall: 0.76


The model performs well in identifying high-risk days (recall = 76%), which is critical for public safety forecasting.

📈 AQI Trend Visualization
Delhi AQI trend from 2015 to 2025:

📊 Model Output Snapshot
High-Risk Forecast System Output:

⚠️ Smart Advisory System

Example Output:
Forecast: HIGH RISK Tomorrow ⚠
Advice: Avoid outdoor exposure.


The system automatically generates health advice based on predicted risk levels.

🛠️ Tech Stack

Python 3.14
Pandas
NumPy
Scikit-Learn
Matplotlib
GitHub

📂 Project Structure

AQI-Air-Quality-Intelligence-System/
│
├── aqi_data.csv
├── explore_data.py
├── clean_data.py
├── train_model.py
├── final_aqi_project.py
├── aqi_trend_forecast.py
├── images/
│   ├── model_output.png
│   └── aqi_trend.png
└── README.md

🎯 Key Features

✔ End-to-end ML pipeline
✔ Real-world environmental dataset
✔ Binary risk prediction system
✔ Visualization dashboard
✔ Smart advisory engine
✔ Forecast simulation

🔮 Future Improvements

Deep Learning (LSTM) time-series forecasting
Multi-city comparison dashboard
Deployment using Streamlit
Live AQI API integration
Web-based public dashboard

📌 Conclusion

This project demonstrates how Machine Learning can be applied to real-world environmental data to build intelligent public health forecasting systems.
It combines:

Data Analysis
Predictive Modeling
Visualization
Decision Support
to create a complete Air Quality Intelligence System.
