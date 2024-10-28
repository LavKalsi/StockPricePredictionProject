# Stock Price Prediction App

A web application that predicts stock price movements for the next five business days using historical data. The frontend is built with Streamlit, and the backend leverages machine learning models (XGBoost and LSTM) for predictions.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Backend Details](#backend-details)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The **Stock Price Prediction App** is a web application that provides short-term price predictions for selected stocks. It uses machine learning models to analyze historical stock data and project future price movements.

## Features

- **Interactive Interface**: Users can select a stock and view predictions for the next five business days.
- **Data Visualization**: Displays recent stock prices along with the forecasted trend.
- **Ensemble Prediction**: Combines XGBoost and LSTM predictions for enhanced accuracy.

## Installation

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/LavKalsi/StockPricePredictionApp.git
    cd StockPricePredictionApp
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the application:
    ```sh
    streamlit run app.py
    ```

## Usage

1. Open your browser and navigate to `http://localhost:8501`.
2. Select a stock ticker from the dropdown menu.
3. View the predicted stock movement for the next five days.

## How It Works

The app uses a machine learning ensemble to predict stock prices:

1. **Data Download**: Historical stock data is fetched from Yahoo Finance.
2. **Preprocessing**: Technical indicators like **SMA**, **EMA**, **RSI**, and **Bollinger Bands** are added.
3. **Prediction**: The processed data is passed to XGBoost and LSTM models. The ensemble output is a combined prediction of both models.
4. **Visualization**: Recent prices and future predictions are plotted on an interactive chart.

## Backend Details

The backend consists of machine learning models and a preprocessing pipeline:

- `xgboost_stock_model.pkl`: Trained XGBoost model for stock prediction.
- `lstm_stock_model.h5`: Trained LSTM model for sequential stock data.
- `scaler.pkl`: Data scaler used for normalization.
- `app.py`: Streamlit app code.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -am 'Add your feature'`).
4. Push the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

LavKalsi - [GitHub](https://github.com/LavKalsi)

Feel free to reach out for questions or suggestions!
