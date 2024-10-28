import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import pickle
from langchain_community.chat_models import ChatPerplexity
from dotenv import load_dotenv
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import torch

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

app = Flask(__name__)


class SalesPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        self.scaler = StandardScaler()
        self.chat_model = ChatPerplexity(
            model="llama-3.1-sonar-large-128k-online",
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
            model_path = os.path.join('models', 'sales_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'features': feature_cols
                }, f)

            return {'metrics': metrics, 'insights': insights}

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

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
        model_path = os.path.join('models', 'sales_model.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        predictor = SalesPredictor()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']

        # Process data and make predictions
        processed_data = predictor.create_features(df)
        X = processed_data[model_data['features']]
        X_scaled = predictor.scaler.transform(X)
        predictions = predictor.model.predict(X_scaled)

        # Generate visualization
        plot_path = predictor.visualize_results(
            actual=df['sales'].values,
            predicted=predictions,
            dates=df['date']
        )

        # Generate AI insights
        insights = predictor.generate_ai_insights(df)

        return jsonify({
            'predictions': predictions.tolist(),
            'plot_url': f'/{plot_path}',
            'insights': insights
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)