import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import pickle
import datetime
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up Perplexity API
os.environ["PPLX_API_KEY"] = os.getenv("PPLX_API_KEY")

app = Flask(__name__)


class SalesPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.chat_model = ChatPerplexity(
            model="llama-3.1-sonar-large-128k-online",
            temperature=0.7
        )

    def generate_ai_insights(self, sales_data: pd.DataFrame) -> str:
        """Generate AI-powered insights from sales data using Perplexity"""
        try:
            sales_summary = {
                'total_sales': sales_data['sales'].sum(),
                'avg_sales': sales_data['sales'].mean(),
                'sales_growth': sales_data['sales'].pct_change().mean() * 100
            }

            # Create prompt template for sales analysis
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a sales analysis expert. Analyze the sales data and provide strategic insights."),
                ("human", f"""
                Analyze this sales data and provide strategic insights:
                Total Sales: ${sales_summary['total_sales']:,.2f}
                Average Sales: ${sales_summary['avg_sales']:,.2f}
                Sales Growth: {sales_summary['sales_growth']:.2f}%

                Provide:
                1. Key trends analysis
                2. Business recommendations
                3. Risk factors
                4. Growth opportunities
                """)
            ])

            # Create chain and invoke
            chain = prompt | self.chat_model
            response = chain.invoke({})

            return response.content

        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return "Unable to generate AI insights at this time."

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering with advanced metrics"""
        df = df.copy()

        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        # Sales metrics
        for window in [7, 14, 30]:
            df[f'sales_ma_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).mean())

        return df

    def train_model(self, train_data: pd.DataFrame) -> dict:
        """Train the model and generate insights"""
        try:
            processed_data = self.create_features(train_data)

            feature_cols = [col for col in processed_data.columns
                            if col not in ['date', 'sales']]

            X = processed_data[feature_cols]
            y = processed_data['sales']

            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)

            # Generate predictions for training data
            y_pred = self.model.predict(X_scaled)

            # Calculate metrics
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }

            # Generate AI insights
            insights = self.generate_ai_insights(processed_data)

            # Save model
            with open('sales_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'features': feature_cols
                }, f)

            return {'metrics': metrics, 'insights': insights}

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/train', methods=['POST'])
def train():
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'])

        predictor = SalesPredictor()
        result = predictor.train_model(df)

        return jsonify({
            'message': 'Model trained successfully',
            'metrics': result['metrics'],
            'insights': result['insights']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'])

        # Load model
        with open('sales_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        predictor = SalesPredictor()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']

        # Process data and make predictions
        processed_data = predictor.create_features(df)
        X = processed_data[model_data['features']]
        X_scaled = predictor.scaler.transform(X)
        predictions = predictor.model.predict(X_scaled)

        # Generate AI insights
        insights = predictor.generate_ai_insights(df)

        return jsonify({
            'predictions': predictions.tolist(),
            'insights': insights
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)