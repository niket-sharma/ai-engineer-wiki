---
title: "Exxon"
aliases: ["ExxonMobil AI", "Exxon  prep", "ExxonMobil", "XOM"]
tags: [company, energy, optimization, timeseries, industrial-ai]
related: ["[[ml-platform]]"]
sources: ["training-knowledge", "public-job-postings"]
relevance: high
last_updated: 2026-04-22
status: current
---

# Exxon (ExxonMobil)

## Role Focus
ExxonMobil has a significant AI/ML function (ExxonMobil Technology & Engineering). Focus areas:
- **Predictive maintenance**: sensor data from refineries and upstream equipment, anomaly detection
- **Process optimization**: optimize chemical plant operations using ML + simulation
- **Commodity pricing**: time series forecasting for oil/gas prices, supply-demand modeling
- **Seismic interpretation**: geological data for exploration (computer vision on seismic images)
- **Energy transition**: ML for carbon capture, hydrogen production optimization

AI/ML roles here are more "industrial AI" than "LLM" — expect more emphasis on time series, signal processing, optimization, and domain physics.

## Tech Stack Signals
- **Cloud**: Azure primary (Microsoft partnership)
- **ML**: Python, PyTorch, scikit-learn
- **Domain tools**: MATLAB, AVEVA (process simulation), OSIsoft PI (industrial time series)
- **Data**: industrial IoT sensor data, time series databases (InfluxDB or PI)
- **Scale**: smaller data volumes than fintech but high-stakes reliability requirements

##  Style
- **Applied ML depth**: expect questions grounded in industrial use cases
- **Time series**: anomaly detection, forecasting, signal processing will feature heavily
- **Optimization**: linear programming, reinforcement learning for process control
- **Less LeetCode-heavy** than finance companies — more applied ML problem solving
- **Domain curiosity**: understanding why the physics matters (what is a refinery doing?) is valued

## Domain-Specific Concepts

### Predictive Maintenance
- **Problem**: predict equipment failure before it happens using sensor data
- **Approaches**:
  - **Anomaly detection**: flag unusual sensor readings (IsolationForest, LSTM-Autoencoder, statistical control charts)
  - **Remaining Useful Life (RUL)**: regression on time-to-failure (CMAPSS dataset is the benchmark)
  - **Survival analysis**: time-to-event modeling for equipment failure
- **Data challenges**: sensors generate high-frequency multivariate time series with noise, missing readings
- **False positive cost**: shutting down equipment unnecessarily is expensive → precision over recall

### Process Optimization
- **Control problem**: adjust input variables (temperature, pressure, flow rate) to maximize output quality/yield
- **Model-based**: learn a surrogate model of the process, then optimize inputs using gradient descent or BO
- **Reinforcement learning**: treat the plant as an environment, train agent to control setpoints
- **Bayesian Optimization**: when simulations are expensive, optimize with few evaluations

### Commodity Pricing (Time Series)
- **Oil price drivers**: geopolitical events, OPEC decisions, inventory levels, USD strength
- **Models**: ARIMA/SARIMA (baseline), LSTM/Transformer for deep learning, XGBoost with lag features
- **Non-stationarity**: oil prices are not stationary — need differencing or percentage returns
- **Regime changes**: price series has structural breaks (2014 oil crash, COVID) — handle with HMM or changepoint detection

### Seismic Interpretation (Computer Vision)
- **Task**: classify rock formations from 3D seismic amplitude volumes
- **Models**: 3D CNNs, U-Net for segmentation
- **Data**: very expensive to label (requires geologist expertise)

## Likely Questions
1. "How would you build an anomaly detection system for refinery sensor data?"
2. "Walk me through time series forecasting for commodity prices. What models and preprocessing?"
3. "How would you apply reinforcement learning to chemical process optimization?"
4. "Describe Bayesian Optimization and when you'd use it over gradient descent."
5. "How do you handle missing values in high-frequency sensor data?"
6. "What is Remaining Useful Life (RUL) prediction and how would you model it?"
7. "How do you deal with concept drift in a time series model for energy prices?"

## Red Flags to Avoid
- Don't propose LLM-first solutions — industrial AI at Exxon is more classical ML + domain physics
- Don't ignore sensor data preprocessing (noise, missing values, synchronization) — interviewers know this is hard
- Don't ignore the cost asymmetry: unplanned downtime is catastrophic; false alarms are expensive but manageable

## Connections
- [[ml-platform]] — ML infrastructure for training and deploying industrial models
