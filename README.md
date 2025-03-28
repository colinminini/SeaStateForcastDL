# Sea State Forecasting with Machine Learning & Deep Learning
This project focuses on forecasting marine wave height (WaveHeight) using time series models trained on a high-resolution dataset spanning almost two decades of hourly oceanographic data. The goal is to predict the next 24 hours of wave height using the previous 24 hours of all available features.

🗂️ Dataset Overview
Source: Internal Marine Dataset (confidential)

Total Samples: 122,230 hourly entries

Time Range: 2006-09-25 to 2025-03-28

Resolution: 1-hour

Missing Years: 2008 and 2009 (excluded due to excessive missing data)

Preprocessing:

Time-based linear interpolation for missing values (method='time')

Sliding context windows of fixed size for sequence modeling

Validation Split:

Years 2022, 2023, 2024, 2025

🔍 Modeling Approach
✅ Problem Framing
Input: Past 24 hours of all features

Target: Next 24 hours of WaveHeight

Multi-step, direct forecast

🧠 Models Used
Category	Models
Traditional ML	XGBoost, Random Forest
Deep Learning	LSTM, TCN (Temporal Convolutional Network)
📊 Key Findings
Context Windows:

Overlapping context windows (with step < window size) clearly outperform disjoint or full overlap.

Larger context windows help LSTM and TCN, but don't impact XGBoost/RF significantly.

Performance:

XGBoost shines in short-term forecasts (first 10 hours) due to its reactivity.

DL models perform better in the longer-term thanks to their smoother forecast profiles.

Output Characteristics:

Deep learning forecasts are smooth and stable.

XGBoost predictions are noisier, which helps in the short term but hurts long-term accuracy.

