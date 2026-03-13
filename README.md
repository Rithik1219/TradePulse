# TradePulse
# TradePulse: Event-Driven Financial Sentiment & Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20LSTM-orange)
![Data Streaming](https://img.shields.io/badge/Data%20Streaming-Apache%20Kafka-red)

## 📌 Project Overview
TradePulse is an end-to-end, event-driven algorithmic trading and portfolio monitoring system. It is designed to ingest live market data, process unstructured financial news in real-time using Large Language Models (LLMs), and feed structured sentiment scores into a custom Hybrid Ensemble machine learning model to predict short-term asset volatility.

Unlike standard predictive models that suffer from the curse of dimensionality on noisy financial data, this system utilizes a specialized architecture that isolates sequence processing (LSTMs) and tabular data processing (XGBoost) before fusing them via a Meta-Learner.

## 🏗️ System Architecture 

The pipeline is broken down into modular, decoupled components:

1. **Portfolio Ingestion:** Authenticates via the Angel One SmartAPI to track live positions.
2. **Real-Time Event Stream:** Python web scrapers monitor financial news and act as Apache Kafka Producers, streaming breaking news articles without bottlenecking the system.
3. **Unstructured Data Processing:** A Kafka Consumer routes text to an LLM (Gemini/OpenAI) via strict prompt engineering to extract numerical sentiment and urgency scores.
4. **Hybrid Predictive Engine:**
   - **Dimensionality Reduction:** PCA compresses thousands of raw technical indicators.
   - **Sequence Engine:** An LSTM neural network evaluates sentiment momentum.
   - **Tabular Engine:** An XGBoost model evaluates the compressed technical setup.
   - **Meta-Learner:** A final logistic layer determines the ultimate directional probability.
5. **Alerting System:** A Django backend manages historical data and triggers real-time webhook alerts (Discord/Telegram) when high-probability setups are detected.

## 💻 Tech Stack
* **Languages:** Python
* **Machine Learning:** `scikit-learn`, `xgboost`, `PyTorch`
* **Data Engineering:** Apache Kafka, `pandas`, `BeautifulSoup4`
* **APIs & Integration:** Angel One SmartAPI, OpenAI/Gemini API
* **Backend:** Django

## 🚀 Current Status /
