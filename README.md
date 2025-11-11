# 📊 Demand Forecasting ML

> **ML-powered demand forecasting system for supply chain optimization using LSTM and Prophet models with interactive visualizations**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-FF6F00.svg)](https://www.tensorflow.org/)
[![Prophet](https://img.shields.io/badge/Prophet-Latest-brightgreen.svg)](https://facebook.github.io/prophet/)

## 🎯 Overview

This project implements advanced machine learning models for **demand forecasting** in supply chain environments. It combines the power of **LSTM (Long Short-Term Memory)** neural networks and **Facebook Prophet** for time series prediction, enabling businesses to:

- 📈 Predict future product demand with high accuracy
- 🔄 Optimize inventory levels and reduce stockouts
- 💰 Minimize holding costs and improve cash flow
- 📊 Visualize trends, seasonality, and forecast confidence intervals
- ⚡ Generate forecasts for multiple products and time horizons

## ✨ Key Features

- **Dual Model Architecture**: Implements both LSTM and Prophet models for comparison
- **Automatic Feature Engineering**: Creates lag features, rolling means, and exponentially weighted features
- **Synthetic Data Generation**: Built-in function to generate realistic demand data with trends and seasonality
- **Model Evaluation**: Comprehensive metrics (MAE, RMSE, R²) for performance assessment
- **Interactive Visualizations**: Beautiful plots showing historical data, predictions, and confidence intervals
- **Scalable Design**: Object-oriented implementation for easy extension and customization

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
tensorflow >= 2.8.0
prophet
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### Installation

1. Clone this repository:
```bash
git clone https://github.com/sksonip/demand-forecasting-ml.git
cd demand-forecasting-ml
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from forecast_model import DemandForecaster

# Initialize forecaster
forecaster = DemandForecaster()

# Generate sample data (or load your own)
forecaster.generate_sample_data(days=730)

# Train LSTM model
forecaster.prepare_lstm_data(lookback=30)
forecaster.build_lstm_model(units=50)
forecaster.train_lstm(epochs=50, batch_size=32)

# Generate predictions
forecaster.predict_lstm()

# Train Prophet model
forecaster.train_prophet()
forecaster.predict_prophet(periods=90)

# Evaluate and visualize
metrics = forecaster.evaluate_models()
forecaster.plot_results('forecast_results.png')
```

## 📁 Project Structure

```
demand-forecasting-ml/
│
├── forecast_model.py      # Main forecasting implementation
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── LICENSE               # MIT License
└── .gitignore           # Git ignore rules
```

## 🔬 Models Explained

### LSTM (Long Short-Term Memory)

- **Architecture**: 2 LSTM layers with dropout for regularization
- **Input**: 30-day lookback window of historical demand
- **Output**: Next-day demand prediction
- **Advantages**: Captures complex non-linear patterns and long-term dependencies

### Prophet

- **Components**: Trend, yearly/weekly/daily seasonality
- **Mode**: Multiplicative seasonality for better handling of varying amplitude
- **Advantages**: Robust to missing data, handles outliers, easy to interpret

## 📊 Example Output

The system generates comprehensive visualizations including:

1. **Historical Demand**: Time series plot of actual demand
2. **LSTM Predictions**: Comparison of predicted vs actual values
3. **Prophet Forecast**: Future predictions with confidence intervals
4. **Performance Metrics**: MAE, RMSE, and R² scores

## 🎯 Use Cases

- **Retail**: Forecast product demand for inventory planning
- **E-commerce**: Predict seasonal sales patterns
- **Manufacturing**: Optimize production schedules
- **Logistics**: Plan warehouse capacity and distribution
- **Supply Chain**: Reduce bullwhip effect and improve coordination

## 📈 Performance

Typical performance on historical data:
- **LSTM Model**: R² > 0.85, RMSE < 10% of mean demand
- **Prophet Model**: MAPE < 15% for 30-day horizon

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Satish Kumar Soni**
- GitHub: [@sksonip](https://github.com/sksonip)
- LinkedIn: [sksonip](https://linkedin.com/in/sksonip)
- Portfolio: [sksonip.github.io](https://sksonip.github.io)

## 🙏 Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Facebook for the Prophet forecasting tool
- Supply chain analytics community for inspiration

---

<div align="center">
  <p><strong>⭐ If you find this project useful, please consider giving it a star! ⭐</strong></p>
</div>
