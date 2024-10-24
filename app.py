from fastapi import FastAPI, HTTPException
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

app = FastAPI()

# Define the root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Industry and Stock Analysis API!"}

# Load Data function
def load_data():
    iip_data = pd.read_excel('IIP2024.xlsx')
    synthetic_data = pd.read_excel('Synthetic_Industry_Data.xlsx', sheet_name=None)

    stock_data = {}
    stock_data_folder = 'stockdata'
    for filename in os.listdir(stock_data_folder):
        if filename.endswith('.csv'):
            stock_name = filename.replace('.csv', '')
            stock_data[stock_name] = pd.read_csv(os.path.join(stock_data_folder, filename))

    correlation_results = pd.read_excel(os.path.join(stock_data_folder, 'Manufacture_of_Food_Products_correlation_results.xlsx'))

    financial_data = {}
    financial_folder = 'financial'
    for filename in os.listdir(financial_folder):
        if filename.endswith('.xlsx'):
            stock_name = filename.replace('.xlsx', '')
            stock_file_path = os.path.join(financial_folder, filename)
            financial_data[stock_name] = pd.read_excel(stock_file_path, sheet_name=None)

    return iip_data, synthetic_data, stock_data, correlation_results, financial_data

iip_data, synthetic_data, stock_data, correlation_results, financial_data = load_data()

# Define Industry Indicators
indicators = {
    'Manufacture of Food Products': {
        'Leading': ['Consumer Spending Trends', 'Agricultural Output', 'Retail Sales Data'],
        'Lagging': ['Inventory Levels', 'Employment Data']
    },
    'Manufacture of Beverages': {
        'Leading': ['Consumer Confidence', 'Raw Material Prices'],
        'Lagging': ['Production Output', 'Profit Margins']
    },
    # Add other industries similarly...
}

# Helper function to interpret correlation
def interpret_correlation(value):
    if value > 0.8:
        return "Strong Positive"
    elif 0.3 < value <= 0.8:
        return "Slight Positive"
    elif -0.3 <= value <= 0.3:
        return "Neutral"
    elif -0.8 <= value < -0.3:
        return "Slight Negative"
    else:
        return "Strong Negative"

# Prediction Function
def predict_models(industry_name):
    normalized_industry = industry_name.strip().lower()
    matched_sheet_name = None
    
    for sheet_name in synthetic_data.keys():
        if sheet_name.strip().lower() == normalized_industry:
            matched_sheet_name = sheet_name
            break
    
    if not matched_sheet_name:
        raise HTTPException(status_code=404, detail="Industry not found")

    industry_data = synthetic_data[matched_sheet_name]

    # Prepare Data for Modeling
    leading_indicators = indicators[industry_name]['Leading']
    X = industry_data[leading_indicators].shift(1).dropna()
    y = iip_data[industry_name].loc(X.index)

    # Linear Regression Model
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    reg_pred = reg_model.predict(X)

    # ARIMA Model
    arima_model = ARIMA(y, order=(5, 1, 0))  # Adjust the ARIMA order as per your dataset
    arima_result = arima_model.fit()
    arima_pred = arima_result.predict(start=1, end=len(y), dynamic=False)

    # RandomForest Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_pred = rf_model.predict(X)

    # Calculate RMSE for each model
    reg_rmse = mean_squared_error(y, reg_pred, squared=False)
    arima_rmse = mean_squared_error(y, arima_pred, squared=False)
    rf_rmse = mean_squared_error(y, rf_pred, squared=False)

    return {
        "Linear Regression RMSE": reg_rmse,
        "ARIMA RMSE": arima_rmse,
        "RandomForest RMSE": rf_rmse,
        "Linear Regression Prediction": reg_pred.tolist(),
        "ARIMA Prediction": arima_pred.tolist(),
        "RandomForest Prediction": rf_pred.tolist()
    }

# Get financial data for stock
def get_latest_financial_data(stock_name):
    if stock_name in financial_data:
        stock_financial_data = financial_data[stock_name]
        balance_sheet = stock_financial_data.get('BalanceSheet', pd.DataFrame())
        income_statement = stock_financial_data.get('IncomeStatement', pd.DataFrame())
        cash_flow = stock_financial_data.get('CashFlow', pd.DataFrame())

        latest_balance_sheet = balance_sheet.iloc[-1] if not balance_sheet.empty else {}
        latest_income_statement = income_statement.iloc[-1] if not income_statement.empty else {}
        latest_cash_flow = cash_flow.iloc[-1] if not cash_flow.empty else {}

        return {
            "Balance Sheet": latest_balance_sheet.to_dict(),
            "Income Statement": latest_income_statement.to_dict(),
            "Cash Flow": latest_cash_flow.to_dict()
        }
    else:
        raise HTTPException(status_code=404, detail="Stock not found")

# API Endpoints

@app.get("/predict/{industry}")
def get_prediction(industry: str):
    try:
        predictions = predict_models(industry)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/financial/{stock}")
def get_financial(stock: str):
    try:
        financial_data = get_latest_financial_data(stock)
        return financial_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/correlation/{stock}")
def get_stock_correlation(stock: str):
    stock_correlation_data = correlation_results[correlation_results['Stock Name'] == stock]
    if not stock_correlation_data.empty:
        stock_correlation_data['Interpreted'] = stock_correlation_data.apply(
            lambda row: interpret_correlation(row['correlation with Total Revenue/Income']), axis=1
        )
        return stock_correlation_data.to_dict(orient='records')
    else:
        raise HTTPException(status_code=404, detail="Stock not found")

