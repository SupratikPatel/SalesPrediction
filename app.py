# Standard libraries
import os
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from datetime import datetime

# Data processing and ML
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Visualization
import matplotlib.pyplot as plt

# Deep Learning
import torch

# FastAPI
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Other utilities
import pickle
from langchain_community.chat_models import ChatPerplexity
from dotenv import load_dotenv

# Configure logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# GPU Configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    logger.info("GPU not available, using CPU")

# Set up Perplexity API
os.environ["PPLX_API_KEY"] = os.getenv("PPLX_API_KEY")

# Initialize FastAPI
app = FastAPI(title="Sales Prediction System")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

class SalesPredictor:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror',
            tree_method='gpu_hist' if torch.cuda.is_available() else 'hist'
        )
        self.scaler = StandardScaler()
        self.chat_model = ChatPerplexity(
            model="llama-3.1-sonar-small-128k-online",
            temperature=0.7
        )

    def create_supervised_data(self, data, lag=12):
        """Create supervised learning dataset with lagged features"""
        df = pd.DataFrame(data)
        columns = [df.shift(i) for i in range(1, lag + 1)]
        columns.append(df)
        df = pd.concat(columns, axis=1)
        df.fillna(0, inplace=True)
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering with parallel processing"""
        df = df.copy()

        # Parallel processing for time-based features
        with ThreadPoolExecutor() as executor:
            df['year'] = executor.submit(lambda x: x.dt.year, df['date']).result()
            df['month'] = executor.submit(lambda x: x.dt.month, df['date']).result()
            df['quarter'] = executor.submit(lambda x: x.dt.quarter, df['date']).result()
            df['day_of_week'] = executor.submit(lambda x: x.dt.dayofweek, df['date']).result()

        # Create sales difference features
        df['sales_diff'] = df['sales'].diff()

        # Create supervised data
        sales_supervised = self.create_supervised_data(df['sales_diff'])

        # Sales metrics using vectorized operations
        for window in [7, 14, 30]:
            df[f'sales_ma_{window}'] = (
                df.groupby(['store', 'item'])['sales']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[f'sales_std_{window}'] = (
                df.groupby(['store', 'item'])['sales']
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )

        return df

    def train_model(self, train_data: pd.DataFrame) -> Dict:
        """Train the model and generate insights"""
        try:
            processed_data = self.create_features(train_data)

            # Use the same approach as the original code that worked
            feature_cols = [col for col in processed_data.columns
                            if col not in ['date', 'sales']]

            X = processed_data[feature_cols]
            y = processed_data['sales']

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Modify XGBoost parameters to remove early_stopping if dataset is too small
            if len(X) < 100:  # For small datasets
                self.model = XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=7,
                    random_state=42,
                    n_jobs=-1,
                    objective='reg:squarederror',
                    tree_method='gpu_hist' if torch.cuda.is_available() else 'hist'
                )
                # Train without early stopping
                self.model.fit(X_scaled, y)
            else:
                # For larger datasets, use validation set and early stopping
                train_size = int(len(processed_data) * 0.8)
                X_train = X_scaled[:train_size]
                y_train = y[:train_size]
                X_valid = X_scaled[train_size:]
                y_valid = y[train_size:]

                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    verbose=True
                )

            # Generate predictions
            y_pred = self.model.predict(X_scaled)

            # Calculate metrics
            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                'mae': float(mean_absolute_error(y, y_pred)),
                'r2': float(r2_score(y, y_pred))
            }

            # Generate AI insights
            insights = self.generate_ai_insights(processed_data)

            # Save model
            os.makedirs('models', exist_ok=True)
            with open('models/sales_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'features': feature_cols
                }, f)

            return {'metrics': metrics, 'insights': insights}

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def generate_ai_insights(self, sales_data: pd.DataFrame) -> str:
        """Generate AI-powered insights from sales data with better formatting"""
        try:
            sales_summary = {
                'total_sales': sales_data['sales'].sum(),
                'avg_sales': sales_data['sales'].mean(),
                'sales_growth': sales_data['sales'].pct_change().mean() * 100,
                'peak_sales': sales_data['sales'].max(),
                'lowest_sales': sales_data['sales'].min()
            }

            prompt = f"""
            Analyze this sales data and provide strategic insights in a well-formatted way:

            Sales Summary:
            • Total Sales: ${sales_summary['total_sales']:,.2f}
            • Average Sales: ${sales_summary['avg_sales']:,.2f}
            • Sales Growth: {sales_summary['sales_growth']:.2f}%
            • Peak Sales: ${sales_summary['peak_sales']:,.2f}
            • Lowest Sales: ${sales_summary['lowest_sales']:,.2f}

            Please provide insights in the following format:

            1. Key Trends:
            • [Trend 1]
            • [Trend 2]
            • [Trend 3]

            2. Business Recommendations:
            • [Recommendation 1]
            • [Recommendation 2]
            • [Recommendation 3]

            3. Risk Factors:
            • [Risk 1]
            • [Risk 2]
            • [Risk 3]

            4. Growth Opportunities:
            • [Opportunity 1]
            • [Opportunity 2]
            • [Opportunity 3]

            Format each section with bullet points and clear headers.
            Keep each point concise and actionable.
            """

            response = self.chat_model.invoke(prompt)
            formatted_insights = response.content

            # Add HTML formatting
            formatted_insights = formatted_insights.replace('1. Key Trends:',
                                                            '<h4 class="insight-header">Key Trends:</h4>')
            formatted_insights = formatted_insights.replace('2. Business Recommendations:',
                                                            '<h4 class="insight-header">Business Recommendations:</h4>')
            formatted_insights = formatted_insights.replace('3. Risk Factors:',
                                                            '<h4 class="insight-header">Risk Factors:</h4>')
            formatted_insights = formatted_insights.replace('4. Growth Opportunities:',
                                                            '<h4 class="insight-header">Growth Opportunities:</h4>')

            # Convert bullet points to HTML
            formatted_insights = formatted_insights.replace('• ', '<li>')
            formatted_insights = formatted_insights.replace('\n', '</li>\n')

            # Wrap in proper HTML structure
            formatted_insights = f"""
            <div class="insights-container">
                <h3 class="insights-title">AI-Generated Sales Insights</h3>
                <div class="insights-content">
                    {formatted_insights}
                </div>
            </div>
            """

            return formatted_insights

        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return "Unable to generate AI insights at this time."

    def visualize_results(self, actual: np.array, predicted: np.array, dates: pd.Series) -> str:
        """
        Create and save visualization of actual vs predicted sales
        """
        try:
            # Clear any existing plots
            plt.clf()

            # Create the plot
            plt.figure(figsize=(15, 7))

            # Plot actual sales with styling
            plt.plot(dates, actual,
                     label='Actual Sales',
                     marker='o',
                     linestyle='-',
                     color='#2E86C1',
                     markersize=4,
                     alpha=0.8)

            # Plot predicted sales with styling
            plt.plot(dates, predicted,
                     label='Predicted Sales',
                     marker='x',
                     linestyle='--',
                     color='#E74C3C',
                     markersize=4,
                     alpha=0.8)

            # Customize the plot
            plt.title('Sales Prediction Analysis', fontsize=14, pad=20)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Sales Volume', fontsize=12)

            # Add grid for better readability
            plt.grid(True, linestyle='--', alpha=0.3)

            # Customize legend
            plt.legend(fontsize=10, loc='upper left')

            # Rotate x-axis labels
            plt.xticks(rotation=45)

            # Add subtle background grid
            plt.grid(True, linestyle='--', alpha=0.2)

            # Adjust layout
            plt.tight_layout()

            # Ensure static directory exists
            os.makedirs('static', exist_ok=True)

            # Save plot
            plot_path = 'static/prediction_plot.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            return plot_path

        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            raise


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# FastAPI routes:

@app.post("/train")
async def train(file: UploadFile = File(...)):
    start_time = datetime.now()
    logger.info(f"Training request received at {start_time}")

    try:
        if not file:
            return JSONResponse(
                status_code=400,
                content={'error': 'No file uploaded'}
            )
        if not file.filename.endswith('.csv'):
            return JSONResponse(
                status_code=400,
                content={'error': 'File must be CSV format'}
            )

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df['date'] = pd.to_datetime(df['date'])

        predictor = SalesPredictor()
        result = predictor.train_model(df)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"Training completed at {end_time}. Processing time: {processing_time:.2f} seconds")

        return JSONResponse({
            'message': 'Model trained successfully',
            'metrics': result['metrics'],
            'insights': result['insights'],
            'processing_time': f"{processing_time:.2f} seconds"
        })
    except Exception as e:
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.error(f"Training error at {end_time} after {processing_time:.2f} seconds: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = datetime.now()
    logger.info(f"Prediction request received at {start_time}")

    try:
        if not file:
            return JSONResponse(
                status_code=400,
                content={'error': 'No file uploaded'}
            )
        if not file.filename.endswith('.csv'):
            return JSONResponse(
                status_code=400,
                content={'error': 'File must be CSV format'}
            )

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df['date'] = pd.to_datetime(df['date'])

        model_path = os.path.join('models', 'sales_model.pkl')
        if not os.path.exists(model_path):
            return JSONResponse(
                status_code=400,
                content={'error': 'Model not trained. Please train the model first.'}
            )

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        predictor = SalesPredictor()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']

        processed_data = predictor.create_features(df)
        X = processed_data[model_data['features']]
        X_scaled = predictor.scaler.transform(X)
        predictions = predictor.model.predict(X_scaled)

        plot_path = predictor.visualize_results(
            actual=df['sales'].values,
            predicted=predictions,
            dates=df['date']
        )

        insights = predictor.generate_ai_insights(df)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"Prediction completed at {end_time}. Processing time: {processing_time:.2f} seconds")

        return JSONResponse({
            'predictions': predictions.tolist(),
            'plot_url': f'/{plot_path}',
            'insights': insights,
            'processing_time': f"{processing_time:.2f} seconds"
        })
    except Exception as e:
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.error(f"Prediction error at {end_time} after {processing_time:.2f} seconds: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
