import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Food Delivery Analytics Platform",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
        border-bottom: 2px solid #4ECDC4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-card {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions
@st.cache_data
def calculate_distance_vectorized(lat1, lon1, lat2, lon2):
    """Calculate distance using Haversine formula (vectorized)"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    r = 6371  # Earth radius in km
    return c * r

@st.cache_data
def generate_enhanced_sample_data(n_samples=10000):
    """Generate comprehensive sample delivery data with more features"""
    np.random.seed(42)
    
    # Define city centers for realistic location generation
    cities = {
        'Mumbai': (19.0760, 72.8777),
        'Delhi': (28.7041, 77.1025),
        'Bangalore': (12.9716, 77.5946),
        'Hyderabad': (17.3850, 78.4867),
        'Chennai': (13.0827, 80.2707),
        'Kolkata': (22.5726, 88.3639),
        'Pune': (18.5204, 73.8567),
        'Ahmedabad': (23.0225, 72.5714)
    }
    
    # Generate base data
    sample_data = {
        'ID': [f'ORD{i:06d}' for i in range(n_samples)],
        'Delivery_person_ID': [f'DEL{i%500:03d}' for i in range(n_samples)],
        'Restaurant_ID': [f'REST{i%1000:04d}' for i in range(n_samples)],
        'Customer_ID': [f'CUST{i%2000:04d}' for i in range(n_samples)],
    }
    
    # Add temporal features
    start_date = datetime.now() - timedelta(days=365)
    sample_data['Order_Date'] = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
    sample_data['Order_Time'] = [np.random.randint(8, 23) for _ in range(n_samples)]
    sample_data['Day_of_Week'] = [date.weekday() for date in sample_data['Order_Date']]
    sample_data['Month'] = [date.month for date in sample_data['Order_Date']]
    sample_data['Is_Weekend'] = [1 if day >= 5 else 0 for day in sample_data['Day_of_Week']]
    
    # Weather conditions
    sample_data['Weather_Condition'] = np.random.choice(['Clear', 'Rain', 'Cloudy', 'Storm'], 
                                                       n_samples, p=[0.5, 0.2, 0.25, 0.05])
    sample_data['Temperature'] = np.random.normal(28, 5, n_samples)
    
    # Delivery partner details
    sample_data['Delivery_person_Age'] = np.random.randint(18, 60, n_samples)
    sample_data['Delivery_person_Ratings'] = np.random.uniform(3.0, 5.0, n_samples)
    sample_data['Delivery_person_Experience'] = np.random.randint(0, 10, n_samples)
    
    # Location data
    cities_list = list(cities.keys())
    sample_data['City'] = np.random.choice(cities_list, n_samples)
    
    restaurant_lats, restaurant_lons = [], []
    delivery_lats, delivery_lons = [], []
    
    for city in sample_data['City']:
        center_lat, center_lon = cities[city]
        rest_lat = center_lat + np.random.normal(0, 0.05)
        rest_lon = center_lon + np.random.normal(0, 0.05)
        restaurant_lats.append(rest_lat)
        restaurant_lons.append(rest_lon)
        
        del_lat = center_lat + np.random.normal(0, 0.1)
        del_lon = center_lon + np.random.normal(0, 0.1)
        delivery_lats.append(del_lat)
        delivery_lons.append(del_lon)
    
    sample_data['Restaurant_latitude'] = restaurant_lats
    sample_data['Restaurant_longitude'] = restaurant_lons
    sample_data['Delivery_location_latitude'] = delivery_lats
    sample_data['Delivery_location_longitude'] = delivery_lons
    
    # Order details
    sample_data['Type_of_order'] = np.random.choice(['Snack', 'Drinks', 'Meal', 'Buffet', 'Dessert'], n_samples)
    sample_data['Type_of_vehicle'] = np.random.choice(['motorcycle', 'scooter', 'bicycle', 'car'], 
                                                     n_samples, p=[0.4, 0.35, 0.15, 0.1])
    sample_data['Order_Value'] = np.random.uniform(100, 2000, n_samples)
    sample_data['Multiple_Orders'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    sample_data['Peak_Hour'] = [1 if hour in [12, 13, 19, 20, 21] else 0 for hour in sample_data['Order_Time']]
    sample_data['Festival_Day'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    # Restaurant features
    sample_data['Restaurant_Rating'] = np.random.uniform(3.5, 5.0, n_samples)
    sample_data['Restaurant_Prep_Time'] = np.random.randint(5, 30, n_samples)
    sample_data['Cuisine_Type'] = np.random.choice(['Indian', 'Chinese', 'Italian', 'Fast Food', 'Continental'], n_samples)
    
    data = pd.DataFrame(sample_data)
    
    # Calculate distances
    data['distance'] = calculate_distance_vectorized(
        data['Restaurant_latitude'], 
        data['Restaurant_longitude'],
        data['Delivery_location_latitude'], 
        data['Delivery_location_longitude']
    )
    
    # Generate realistic delivery times
    def generate_complex_delivery_time(row):
        base_time = 20
        distance_time = row['distance'] * 2.8
        age_factor = (row['Delivery_person_Age'] - 30) * 0.05
        rating_factor = (5 - row['Delivery_person_Ratings']) * 1.5
        experience_factor = -row['Delivery_person_Experience'] * 0.3
        
        vehicle_factors = {'bicycle': 8, 'scooter': 0, 'motorcycle': -3, 'car': -1}
        vehicle_time = vehicle_factors[row['Type_of_vehicle']]
        
        order_factors = {'Snack': -2, 'Drinks': -3, 'Meal': 2, 'Buffet': 6, 'Dessert': 0}
        order_time = order_factors[row['Type_of_order']]
        prep_time = row['Restaurant_Prep_Time'] * 0.3
        
        peak_hour_delay = row['Peak_Hour'] * 8
        weekend_delay = row['Is_Weekend'] * 3
        festival_delay = row['Festival_Day'] * 10
        
        weather_factors = {'Clear': 0, 'Cloudy': 2, 'Rain': 8, 'Storm': 15}
        weather_delay = weather_factors[row['Weather_Condition']]
        
        multiple_order_delay = row['Multiple_Orders'] * 5
        
        city_factors = {'Mumbai': 5, 'Delhi': 4, 'Bangalore': 6, 'Hyderabad': 2, 
                       'Chennai': 3, 'Kolkata': 4, 'Pune': 2, 'Ahmedabad': 1}
        city_delay = city_factors[row['City']]
        
        noise = np.random.normal(0, 2)
        
        total_time = (base_time + distance_time + age_factor + rating_factor + 
                     experience_factor + vehicle_time + order_time + prep_time +
                     peak_hour_delay + weekend_delay + festival_delay + weather_delay +
                     multiple_order_delay + city_delay + noise)
        
        return max(12, total_time)
    
    data['Time_taken(min)'] = data.apply(generate_complex_delivery_time, axis=1)
    
    # Add customer satisfaction
    def calculate_satisfaction(time_taken):
        if time_taken <= 20:
            return np.random.uniform(4.5, 5.0)
        elif time_taken <= 30:
            return np.random.uniform(3.8, 4.5)
        elif time_taken <= 45:
            return np.random.uniform(3.0, 4.0)
        else:
            return np.random.uniform(2.0, 3.5)
    
    data['Customer_Satisfaction'] = data['Time_taken(min)'].apply(calculate_satisfaction)
    
    return data

class EnhancedDeliveryPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        
    def prepare_features(self, data, fit_encoders=True):
        """Enhanced feature preparation"""
        df = data.copy()
        
        categorical_cols = ['Type_of_order', 'Type_of_vehicle', 'Weather_Condition', 
                           'City', 'Cuisine_Type']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit_encoders:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    if col in self.label_encoders:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Advanced feature engineering
        if 'Delivery_person_Ratings' in df.columns:
            df['rating_category'] = pd.cut(df['Delivery_person_Ratings'], 
                                         bins=[0, 3.5, 4.0, 4.5, 5.0], 
                                         labels=[0, 1, 2, 3])
        
        if 'Delivery_person_Age' in df.columns:
            df['age_group'] = pd.cut(df['Delivery_person_Age'], 
                                   bins=[0, 25, 35, 45, 100], 
                                   labels=[0, 1, 2, 3])
        
        if 'distance' in df.columns:
            df['distance_category'] = pd.cut(df['distance'], 
                                           bins=[0, 3, 7, 15, 100], 
                                           labels=[0, 1, 2, 3])
        
        if 'Order_Value' in df.columns:
            df['order_value_category'] = pd.cut(df['Order_Value'], 
                                               bins=[0, 300, 700, 1200, 10000], 
                                               labels=[0, 1, 2, 3])
        
        # Interaction features
        if 'distance' in df.columns and 'Peak_Hour' in df.columns:
            df['distance_peak_interaction'] = df['distance'] * df['Peak_Hour']
        
        if 'Weather_Condition_encoded' in df.columns and 'distance' in df.columns:
            df['weather_distance_interaction'] = df['Weather_Condition_encoded'] * df['distance']
        
        # Select feature columns
        feature_cols = []
        possible_features = [
            'Delivery_person_Age', 'Delivery_person_Ratings', 'Delivery_person_Experience',
            'distance', 'Order_Time', 'Day_of_Week', 'Is_Weekend', 'Peak_Hour',
            'Temperature', 'Multiple_Orders', 'Festival_Day', 'Order_Value',
            'Restaurant_Rating', 'Restaurant_Prep_Time'
        ]
        
        encoded_features = [col for col in df.columns if col.endswith('_encoded')]
        category_features = [col for col in df.columns if col.endswith('_category')]
        interaction_features = [col for col in df.columns if 'interaction' in col]
        
        for col in possible_features + encoded_features + category_features + interaction_features:
            if col in df.columns:
                feature_cols.append(col)
        
        return df[feature_cols].fillna(0)
    
    def train_models(self, X_train, y_train):
        """Train enhanced models"""
        models_to_train = {
            'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, max_depth=8, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        for name, model in models_to_train.items():
            model.fit(X_train, y_train)
            self.models[name] = model
        
        self.is_fitted = True
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            accuracy_5min = np.mean(np.abs(y_test - y_pred) <= 5) * 100
            accuracy_10min = np.mean(np.abs(y_test - y_pred) <= 10) * 100
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Accuracy_5min': accuracy_5min,
                'Accuracy_10min': accuracy_10min,
                'Predictions': y_pred
            }
        
        return results
    
    def predict(self, features, model_name='Random Forest'):
        """Predict delivery time"""
        if not self.is_fitted:
            raise ValueError("Model needs to be trained first!")
        
        model = self.models[model_name]
        return model.predict(features)

@st.cache_resource
def load_and_train_enhanced_model():
    """Load enhanced data and train models"""
    data = generate_enhanced_sample_data()
    predictor = EnhancedDeliveryPredictor()
    
    X = predictor.prepare_features(data, fit_encoders=True)
    y = data['Time_taken(min)'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    predictor.train_models(X_train_scaled, y_train)
    results = predictor.evaluate_models(X_test_scaled, y_test)
    
    return data, predictor, X_test_scaled, y_test, results

def main():
    st.markdown('<h1 class="main-header">🍕 Advanced Food Delivery Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Load data and model
    with st.spinner("🚀 Loading comprehensive data and training advanced models..."):
        data, predictor, X_test, y_test, results = load_and_train_enhanced_model()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    main_pages = {
        "🔮 Smart Prediction": "prediction",
        "📊 Advanced Analytics": "analytics", 
        "📈 ML Performance": "performance",
        "🎯 Batch Processing": "batch",
        "🗺️ Geographic Analysis": "geographic",
        "⏰ Time Analysis": "time",
        "🏪 Restaurant Intelligence": "restaurant",
        "👥 Customer Analytics": "customer",
        "🚚 Delivery Insights": "delivery",
        "🌤️ Weather Impact": "weather",
        "📱 Dashboard": "dashboard",
        "💰 Business Intelligence": "business"
    }
    
    selected_page = st.sidebar.selectbox("Choose Analysis", list(main_pages.keys()))
    page_key = main_pages[selected_page]
    
    # Display selected page
    if page_key == "prediction":
        show_prediction_page(predictor, data)
    elif page_key == "analytics":
        show_analytics_page(data)
    elif page_key == "performance":
        show_performance_page(results, predictor, X_test, y_test, data)
    elif page_key == "batch":
        show_batch_page(predictor)
    elif page_key == "geographic":
        show_geographic_page(data)
    elif page_key == "time":
        show_time_analysis_page(data)
    elif page_key == "restaurant":
        show_restaurant_page(data)
    elif page_key == "customer":
        show_customer_page(data)
    elif page_key == "delivery":
        show_delivery_page(data)
    elif page_key == "weather":
        show_weather_page(data)
    elif page_key == "dashboard":
        show_dashboard_page(data)
    elif page_key == "business":
        show_business_page(data)

def show_prediction_page(predictor, data):
    st.markdown('<h2 class="sub-header">🔮 Smart Delivery Time Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Basic Details")
        age = st.slider("Delivery Partner Age", 18, 65, 30)
        ratings = st.slider("Partner Rating", 1.0, 5.0, 4.2, 0.1)
        experience = st.slider("Experience (years)", 0, 10, 2)
        distance = st.slider("Distance (km)", 0.5, 50.0, 5.0, 0.5)
    
    with col2:
        st.subheader("🚗 Order & Vehicle")
        order_type = st.selectbox("Order Type", ["Snack", "Drinks", "Meal", "Buffet", "Dessert"])
        vehicle_type = st.selectbox("Vehicle", ["motorcycle", "scooter", "bicycle", "car"])
        order_value = st.number_input("Order Value (₹)", 100, 5000, 500)
        cuisine_type = st.selectbox("Cuisine", ["Indian", "Chinese", "Italian", "Fast Food", "Continental"])
    
    with col3:
        st.subheader("🌍 Context")
        city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"])
        weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain", "Storm"])
        temperature = st.slider("Temperature (°C)", 15, 45, 28)
        peak_hour = st.checkbox("Peak Hour")
        weekend = st.checkbox("Weekend")
        festival = st.checkbox("Festival Day")
        multiple_orders = st.checkbox("Multiple Orders")
    
    if st.button("🚀 Predict Delivery Time", type="primary", use_container_width=True):
        try:
            current_time = datetime.now()
            pred_data = pd.DataFrame({
                'Delivery_person_Age': [age],
                'Delivery_person_Ratings': [ratings],
                'Delivery_person_Experience': [experience],
                'distance': [distance],
                'Order_Time': [current_time.hour],
                'Day_of_Week': [current_time.weekday()],
                'Is_Weekend': [1 if weekend else 0],
                'Peak_Hour': [1 if peak_hour else 0],
                'Festival_Day': [1 if festival else 0],
                'Type_of_order': [order_type],
                'Type_of_vehicle': [vehicle_type],
                'Weather_Condition': [weather],
                'Temperature': [temperature],
                'Multiple_Orders': [1 if multiple_orders else 0],
                'Order_Value': [order_value],
                'City': [city],
                'Restaurant_Rating': [4.2],
                'Restaurant_Prep_Time': [15],
                'Cuisine_Type': [cuisine_type],
                'Restaurant_latitude': [12.0],
                'Restaurant_longitude': [77.0],
                'Delivery_location_latitude': [12.0],
                'Delivery_location_longitude': [77.0],
                'Month': [current_time.month]
            })
            
            X_pred = predictor.prepare_features(pred_data, fit_encoders=False)
            X_pred_scaled = predictor.scaler.transform(X_pred)
            
            col1, col2, col3 = st.columns(3)
            predictions = []
            
            for i, model_name in enumerate(predictor.models.keys()):
                pred = predictor.predict(X_pred_scaled, model_name)[0]
                predictions.append(pred)
                
                with [col1, col2, col3][i]:
                    st.markdown(f"""
                    <div class="prediction-result">
                        {model_name}<br>
                        {pred:.1f} min
                    </div>
                    """, unsafe_allow_html=True)
            
            avg_prediction = np.mean(predictions)
            confidence_interval = np.std(predictions)
            
            st.markdown(f"""
            <div class="prediction-result" style="background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);">
                🎯 Ensemble Prediction<br>
                {avg_prediction:.1f} ± {confidence_interval:.1f} minutes
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            estimated_delivery = current_time + timedelta(minutes=int(avg_prediction))
            
            with col1:
                st.metric("📅 Estimated Arrival", estimated_delivery.strftime("%H:%M"))
            with col2:
                if avg_prediction <= 25:
                    st.metric("⚡ Speed", "Fast", "✅")
                elif avg_prediction <= 35:
                    st.metric("🚀 Speed", "Normal", "⚠️")
                else:
                    st.metric("🐌 Speed", "Slow", "❌")
            with col3:
                satisfaction_score = max(1, min(5, 5.5 - avg_prediction/10))
                st.metric("😊 Expected Satisfaction", f"{satisfaction_score:.1f}/5")
            with col4:
                if weather == "Rain":
                    st.metric("🌧️ Weather Impact", "+5-8 min")
                elif weather == "Storm":
                    st.metric("⛈️ Weather Impact", "+10-15 min")
                else:
                    st.metric("☀️ Weather Impact", "Minimal")
                    
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

def show_analytics_page(data):
    st.markdown('<h2 class="sub-header">📊 Advanced Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_delivery_time = data['Time_taken(min)'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>⏱️ Avg Delivery</h3>
            <h2>{avg_delivery_time:.1f} min</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_orders = len(data)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📦 Total Orders</h3>
            <h2>{total_orders:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_satisfaction = data['Customer_Satisfaction'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>😊 Satisfaction</h3>
            <h2>{avg_satisfaction:.1f}/5</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_distance = data['distance'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>📏 Avg Distance</h3>
            <h2>{avg_distance:.1f} km</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        revenue_estimate = (data['Order_Value'].sum() / 1000000)
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Revenue</h3>
            <h2>₹{revenue_estimate:.1f}M</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    tab1, tab2, tab3 = st.tabs(["🎯 Performance Analysis", "🌍 City Comparison", "🔍 Correlations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(data, x="distance", y="Time_taken(min)", 
                            color="Type_of_vehicle", size="Order_Value",
                            title="Distance vs Delivery Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(data, x="Type_of_vehicle", y="Time_taken(min)",
                        title="Delivery Time by Vehicle Type")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        city_stats = data.groupby('City').agg({
            'Time_taken(min)': ['mean', 'std'],
            'Customer_Satisfaction': 'mean',
            'Order_Value': 'mean',
            'ID': 'count'
        }).round(2)
        
        city_stats.columns = ['Avg Time', 'Time StdDev', 'Satisfaction', 'Avg Order Value', 'Total Orders']
        
        st.dataframe(city_stats, use_container_width=True)
        
        fig = px.bar(x=city_stats.index, y=city_stats['Avg Time'],
                    title="Average Delivery Time by City")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        numeric_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance',
                       'Order_Value', 'Temperature', 'Time_taken(min)', 'Customer_Satisfaction']
        
        correlation_data = data[numeric_cols].corr()
        
        fig = px.imshow(correlation_data, title="Feature Correlation Matrix",
                       aspect="auto", color_continuous_scale="RdBu")
        st.plotly_chart(fig, use_container_width=True)

def show_performance_page(results, predictor, X_test, y_test, data):
    st.markdown('<h2 class="sub-header">📈 ML Model Performance</h2>', unsafe_allow_html=True)
    
    # Performance overview
    performance_df = pd.DataFrame({
        'Model': list(results.keys()),
        'RMSE (min)': [results[model]['RMSE'] for model in results.keys()],
        'MAE (min)': [results[model]['MAE'] for model in results.keys()],
        'R² Score': [results[model]['R²'] for model in results.keys()],
        'Accuracy ±5min': [f"{results[model]['Accuracy_5min']:.1f}%" for model in results.keys()],
        'Accuracy ±10min': [f"{results[model]['Accuracy_10min']:.1f}%" for model in results.keys()]
    })
    
    st.dataframe(performance_df, use_container_width=True)
    
    # Best model identification
    best_model = performance_df.loc[performance_df['RMSE (min)'].idxmin(), 'Model']
    best_rmse = performance_df.loc[performance_df['RMSE (min)'].idxmin(), 'RMSE (min)']
    
    st.markdown(f"""
    <div class="success-card">
        <h3>🏆 Best Model: {best_model}</h3>
        <p>RMSE: {best_rmse:.2f} minutes - predictions typically within ±{best_rmse:.1f} minutes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(performance_df, x='Model', y='RMSE (min)',
                    title='RMSE Comparison', color='RMSE (min)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(performance_df, x='Model', y='R² Score',
                    title='R² Score Comparison', color='R² Score')
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction vs Actual
    selected_model = st.selectbox("Select model for analysis:", list(results.keys()))
    y_pred = results[selected_model]['Predictions']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(x=y_test, y=y_pred,
                        labels={'x': 'Actual Time', 'y': 'Predicted Time'},
                        title=f'{selected_model}: Predictions vs Actual')
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                      line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        residuals = y_test - y_pred
        fig = px.histogram(residuals, nbins=30, title="Prediction Error Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance for Random Forest
    if 'Random Forest' in predictor.models:
        st.subheader("🔍 Feature Importance")
        rf_model = predictor.models['Random Forest']
        
        sample_data = generate_enhanced_sample_data(100)
        X_sample = predictor.prepare_features(sample_data)
        feature_names = X_sample.columns
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df.head(10), x='Importance', y='Feature',
                    orientation='h', title='Top 10 Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

def show_batch_page(predictor):
    st.markdown('<h2 class="sub-header">🎯 Batch Processing</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Upload CSV with columns:** Delivery_person_Age, Delivery_person_Ratings, distance, 
    Type_of_order, Type_of_vehicle, Weather_Condition, City, etc.
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded! {len(df)} rows loaded.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("📊 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.write("📈 Data Summary")
                st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            
            if st.button("🚀 Generate Predictions"):
                with st.spinner("Processing..."):
                    try:
                        # Add default values for missing columns
                        default_values = {
                            'Delivery_person_Experience': 2, 'Order_Time': 14,
                            'Is_Weekend': 0, 'Peak_Hour': 0, 'Festival_Day': 0,
                            'Multiple_Orders': 0, 'Temperature': 28,
                            'Restaurant_Rating': 4.2, 'Restaurant_Prep_Time': 15,
                            'Weather_Condition': 'Clear', 'City': 'Bangalore',
                            'Cuisine_Type': 'Indian', 'Order_Value': 500,
                            'Month': 6, 'Day_of_Week': 2
                        }
                        
                        for col, default_val in default_values.items():
                            if col not in df.columns:
                                df[col] = default_val
                        
                        # Calculate distance if coordinates provided
                        if 'distance' not in df.columns:
                            if all(col in df.columns for col in ['Restaurant_latitude', 'Restaurant_longitude', 
                                                                'Delivery_location_latitude', 'Delivery_location_longitude']):
                                df['distance'] = calculate_distance_vectorized(
                                    df['Restaurant_latitude'], df['Restaurant_longitude'],
                                    df['Delivery_location_latitude'], df['Delivery_location_longitude']
                                )
                        
                        X_bulk = predictor.prepare_features(df, fit_encoders=False)
                        X_bulk_scaled = predictor.scaler.transform(X_bulk)
                        
                        # Generate predictions
                        for model_name in predictor.models.keys():
                            predictions = predictor.predict(X_bulk_scaled, model_name)
                            df[f'{model_name}_Prediction'] = predictions.round(1)
                        
                        # Ensemble prediction
                        pred_cols = [col for col in df.columns if 'Prediction' in col]
                        df['Ensemble_Prediction'] = df[pred_cols].mean(axis=1).round(1)
                        
                        st.success(f"✅ Predictions generated for {len(df)} orders!")
                        st.dataframe(df[['Ensemble_Prediction'] + pred_cols].head(), use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name='delivery_predictions.csv',
                            mime='text/csv'
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Processing error: {str(e)}")
                        
        except Exception as e:
            st.error(f"❌ File reading error: {str(e)}")
    
    # Sample data generator
    st.subheader("📝 Generate Sample Data")
    sample_size = st.number_input("Sample size", 10, 1000, 100)
    
    if st.button("🎲 Generate Sample"):
        sample_df = generate_enhanced_sample_data(sample_size)
        sample_columns = [
            'Delivery_person_Age', 'Delivery_person_Ratings', 'distance',
            'Type_of_order', 'Type_of_vehicle', 'Weather_Condition',
            'City', 'Order_Value', 'Peak_Hour', 'Is_Weekend'
        ]
        sample_export = sample_df[sample_columns]
        
        csv = sample_export.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name=f'sample_data_{sample_size}.csv',
            mime='text/csv'
        )

def show_geographic_page(data):
    st.markdown('<h2 class="sub-header">🗺️ Geographic Analysis</h2>', unsafe_allow_html=True)
    
    # City performance comparison
    city_stats = data.groupby('City').agg({
        'Time_taken(min)': ['mean', 'std'],
        'distance': 'mean',
        'Customer_Satisfaction': 'mean',
        'Order_Value': 'mean',
        'ID': 'count'
    }).round(2)
    
    city_stats.columns = ['Avg Time', 'Time StdDev', 'Avg Distance', 'Satisfaction', 'Avg Order Value', 'Total Orders']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("📊 City Performance Summary")
        st.dataframe(city_stats, use_container_width=True)
    
    with col2:
        fig = px.bar(x=city_stats.index, y=city_stats['Avg Time'],
                    title="Average Delivery Time by City",
                    color=city_stats['Satisfaction'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic distribution
    sample_data = data.sample(min(1000, len(data)))
    fig = px.scatter_mapbox(
        sample_data, 
        lat='Restaurant_latitude', 
        lon='Restaurant_longitude',
        color='Time_taken(min)',
        size='Order_Value',
        hover_data=['City', 'Customer_Satisfaction'],
        mapbox_style="open-street-map",
        height=600,
        title="Delivery Performance Geographic Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Distance efficiency analysis
    st.subheader("📏 Distance Efficiency Analysis")
    
    data['efficiency'] = data['distance'] / data['Time_taken(min)']
    efficiency_by_city = data.groupby('City')['efficiency'].mean().sort_values(ascending=False)
    
    fig = px.bar(x=efficiency_by_city.index, y=efficiency_by_city.values,
                title="Delivery Efficiency by City (km/min)")
    st.plotly_chart(fig, use_container_width=True)

def show_time_analysis_page(data):
    st.markdown('<h2 class="sub-header">⏰ Temporal Analysis</h2>', unsafe_allow_html=True)
    
    # Hourly patterns
    hourly_stats = data.groupby('Order_Time').agg({
        'Time_taken(min)': 'mean',
        'ID': 'count',
        'Customer_Satisfaction': 'mean'
    }).round(2)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=hourly_stats.index, y=hourly_stats['ID'], name="Order Count"),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=hourly_stats.index, y=hourly_stats['Time_taken(min)'], 
                  mode='lines+markers', name="Avg Delivery Time"),
        secondary_y=True
    )
    fig.update_xaxes(title_text="Hour of Day")
    fig.update_yaxes(title_text="Order Count", secondary_y=False)
    fig.update_yaxes(title_text="Delivery Time (min)", secondary_y=True)
    fig.update_layout(title_text="Hourly Order Patterns")
    st.plotly_chart(fig, use_container_width=True)
    
    # Daily patterns
    col1, col2 = st.columns(2)
    
    with col1:
        daily_stats = data.groupby('Day_of_Week')['Time_taken(min)'].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = px.bar(x=day_names, y=daily_stats.values,
                    title="Average Delivery Time by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        monthly_stats = data.groupby('Month')['Time_taken(min)'].mean()
        
        fig = px.line(x=monthly_stats.index, y=monthly_stats.values,
                     title="Monthly Delivery Time Trends")
        st.plotly_chart(fig, use_container_width=True)
    
    # Peak hour analysis
    st.subheader("🚀 Peak Hour Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    peak_orders = data[data['Peak_Hour'] == 1]
    regular_orders = data[data['Peak_Hour'] == 0]
    
    with col1:
        st.metric("📊 Peak Hour Orders", f"{len(peak_orders):,}")
        st.metric("⏱️ Avg Peak Delivery", f"{peak_orders['Time_taken(min)'].mean():.1f} min")
    
    with col2:
        st.metric("📊 Regular Orders", f"{len(regular_orders):,}")
        st.metric("⏱️ Avg Regular Delivery", f"{regular_orders['Time_taken(min)'].mean():.1f} min")
    
    with col3:
        time_diff = peak_orders['Time_taken(min)'].mean() - regular_orders['Time_taken(min)'].mean()
        st.metric("⚡ Peak Hour Impact", f"+{time_diff:.1f} min")

def show_restaurant_page(data):
    st.markdown('<h2 class="sub-header">🏪 Restaurant Intelligence</h2>', unsafe_allow_html=True)
    
    # Restaurant performance analysis
    restaurant_stats = data.groupby('Restaurant_ID').agg({
        'Time_taken(min)': ['mean', 'std', 'count'],
        'Customer_Satisfaction': 'mean',
        'Order_Value': 'mean',
        'Restaurant_Rating': 'first',
        'Restaurant_Prep_Time': 'first'
    }).round(2)
    
    restaurant_stats.columns = ['Avg Delivery Time', 'Time StdDev', 'Order Count', 
                               'Customer Satisfaction', 'Avg Order Value', 'Restaurant Rating', 'Prep Time']
    
    # Filter restaurants with sufficient orders
    restaurant_stats = restaurant_stats[restaurant_stats['Order Count'] >= 10]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performing Restaurants")
        top_restaurants = restaurant_stats.nsmallest(10, 'Avg Delivery Time')[['Avg Delivery Time', 'Customer Satisfaction', 'Order Count']]
        st.dataframe(top_restaurants, use_container_width=True)
    
    with col2:
        st.subheader("🐌 Underperforming Restaurants")
        bottom_restaurants = restaurant_stats.nlargest(10, 'Avg Delivery Time')[['Avg Delivery Time', 'Customer Satisfaction', 'Order Count']]
        st.dataframe(bottom_restaurants, use_container_width=True)
    
    # Restaurant rating vs delivery performance
    fig = px.scatter(restaurant_stats, x='Restaurant Rating', y='Avg Delivery Time',
                    size='Order Count', color='Customer Satisfaction',
                    title="Restaurant Rating vs Delivery Performance")
    st.plotly_chart(fig, use_container_width=True)
    
    # Cuisine analysis
    st.subheader("🍽️ Cuisine Performance Analysis")
    
    cuisine_stats = data.groupby('Cuisine_Type').agg({
        'Time_taken(min)': 'mean',
        'Customer_Satisfaction': 'mean',
        'Order_Value': 'mean',
        'ID': 'count'
    }).round(2)
    
    cuisine_stats.columns = ['Avg Delivery Time', 'Satisfaction', 'Avg Order Value', 'Order Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(cuisine_stats, use_container_width=True)
    
    with col2:
        fig = px.bar(cuisine_stats, x=cuisine_stats.index, y='Avg Delivery Time',
                    title="Delivery Time by Cuisine Type")
        st.plotly_chart(fig, use_container_width=True)

def show_customer_page(data):
    st.markdown('<h2 class="sub-header">👥 Customer Analytics</h2>', unsafe_allow_html=True)
    
    # Customer satisfaction analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_satisfaction = data['Customer_Satisfaction'].mean()
        st.metric("😊 Average Satisfaction", f"{avg_satisfaction:.1f}/5")
        
        high_satisfaction = (data['Customer_Satisfaction'] >= 4.0).sum()
        st.metric("👍 High Satisfaction Orders", f"{high_satisfaction:,} ({high_satisfaction/len(data)*100:.1f}%)")
    
    with col2:
        low_satisfaction = (data['Customer_Satisfaction'] <= 3.0).sum()
        st.metric("👎 Low Satisfaction Orders", f"{low_satisfaction:,} ({low_satisfaction/len(data)*100:.1f}%)")
        
        avg_order_value = data['Order_Value'].mean()
        st.metric("💰 Average Order Value", f"₹{avg_order_value:.0f}")
    
    with col3:
        repeat_customers = data.groupby('Customer_ID').size().mean()
        st.metric("🔄 Avg Orders per Customer", f"{repeat_customers:.1f}")
        
        high_value_orders = (data['Order_Value'] >= 1000).sum()
        st.metric("💎 High Value Orders", f"{high_value_orders:,}")
    
    # Satisfaction distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(data, x='Customer_Satisfaction', nbins=20,
                          title="Customer Satisfaction Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(data, x='Order_Value', nbins=30,
                          title="Order Value Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Satisfaction vs delivery time
    fig = px.scatter(data.sample(1000), x='Time_taken(min)', y='Customer_Satisfaction',
                    color='City', title="Satisfaction vs Delivery Time Relationship")
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer segmentation
    st.subheader("🎯 Customer Segmentation")
    
    # Simple segmentation based on order value and frequency
    customer_stats = data.groupby('Customer_ID').agg({
        'Order_Value': 'mean',
        'ID': 'count',
        'Customer_Satisfaction': 'mean'
    }).round(2)
    
    customer_stats.columns = ['Avg Order Value', 'Order Frequency', 'Avg Satisfaction']
    
    # Define segments
    high_value_threshold = customer_stats['Avg Order Value'].quantile(0.75)
    high_frequency_threshold = customer_stats['Order Frequency'].quantile(0.75)
    
    def categorize_customer(row):
        if row['Avg Order Value'] >= high_value_threshold and row['Order Frequency'] >= high_frequency_threshold:
            return 'VIP'
        elif row['Avg Order Value'] >= high_value_threshold:
            return 'High Value'
        elif row['Order Frequency'] >= high_frequency_threshold:
            return 'Frequent'
        else:
            return 'Regular'
    
    customer_stats['Segment'] = customer_stats.apply(categorize_customer, axis=1)
    
    segment_summary = customer_stats.groupby('Segment').agg({
        'Avg Order Value': 'mean',
        'Order Frequency': 'mean',
        'Avg Satisfaction': 'mean'
    }).round(2)
    
    segment_counts = customer_stats['Segment'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Customer Segments Summary:**")
        st.dataframe(segment_summary, use_container_width=True)
    
    with col2:
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title="Customer Segment Distribution")
        st.plotly_chart(fig, use_container_width=True)

def show_delivery_page(data):
    st.markdown('<h2 class="sub-header">🚚 Delivery Partner Insights</h2>', unsafe_allow_html=True)
    
    # Delivery partner performance
    partner_stats = data.groupby('Delivery_person_ID').agg({
        'Time_taken(min)': ['mean', 'std', 'count'],
        'Customer_Satisfaction': 'mean',
        'Delivery_person_Age': 'first',
        'Delivery_person_Ratings': 'first',
        'Delivery_person_Experience': 'first',
        'distance': 'sum'
    }).round(2)
    
    partner_stats.columns = ['Avg Delivery Time', 'Time StdDev', 'Total Deliveries', 
                            'Customer Satisfaction', 'Age', 'Rating', 'Experience', 'Total Distance']
    
    # Filter partners with sufficient deliveries
    partner_stats = partner_stats[partner_stats['Total Deliveries'] >= 10]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performing Partners")
        top_partners = partner_stats.nsmallest(10, 'Avg Delivery Time')[['Avg Delivery Time', 'Customer Satisfaction', 'Total Deliveries', 'Rating']]
        st.dataframe(top_partners, use_container_width=True)
    
    with col2:
        st.subheader("📊 Partner Performance Distribution")
        fig = px.histogram(partner_stats, x='Avg Delivery Time', nbins=20,
                          title="Distribution of Average Delivery Times")
        st.plotly_chart(fig, use_container_width=True)
    
    # Age vs performance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        age_performance = data.groupby('Delivery_person_Age')['Time_taken(min)'].mean()
        fig = px.scatter(x=age_performance.index, y=age_performance.values,
                        title="Age vs Average Delivery Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        rating_performance = data.groupby(pd.cut(data['Delivery_person_Ratings'], bins=5))['Time_taken(min)'].mean()
        fig = px.bar(x=[str(x) for x in rating_performance.index], y=rating_performance.values,
                    title="Rating vs Average Delivery Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Vehicle efficiency
    st.subheader("🚗 Vehicle Efficiency Analysis")
    
    vehicle_stats = data.groupby('Type_of_vehicle').agg({
        'Time_taken(min)': 'mean',
        'Customer_Satisfaction': 'mean',
        'distance': 'mean',
        'ID': 'count'
    }).round(2)
    
    vehicle_stats.columns = ['Avg Delivery Time', 'Satisfaction', 'Avg Distance', 'Order Count']
    vehicle_stats['Efficiency'] = (vehicle_stats['Avg Distance'] / vehicle_stats['Avg Delivery Time']).round(3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(vehicle_stats, use_container_width=True)
    
    with col2:
        fig = px.bar(vehicle_stats, x=vehicle_stats.index, y='Efficiency',
                    title="Vehicle Efficiency (km/min)")
        st.plotly_chart(fig, use_container_width=True)

def show_weather_page(data):
    st.markdown('<h2 class="sub-header">🌤️ Weather Impact Analysis</h2>', unsafe_allow_html=True)
    
    # Weather impact summary
    weather_stats = data.groupby('Weather_Condition').agg({
        'Time_taken(min)': ['mean', 'std'],
        'Customer_Satisfaction': 'mean',
        'ID': 'count'
    }).round(2)
    
    weather_stats.columns = ['Avg Delivery Time', 'Time StdDev', 'Satisfaction', 'Order Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("🌦️ Weather Impact Summary")
        st.dataframe(weather_stats, use_container_width=True)
        
        # Calculate delay vs clear weather
        clear_weather_time = weather_stats.loc['Clear', 'Avg Delivery Time']
        weather_stats['Delay vs Clear'] = weather_stats['Avg Delivery Time'] - clear_weather_time
        
        st.write("⏰ Additional Delay vs Clear Weather")
        st.dataframe(weather_stats[['Delay vs Clear']], use_container_width=True)
    
    with col2:
        fig = px.box(data, x='Weather_Condition', y='Time_taken(min)',
                    title="Delivery Time Distribution by Weather")
        st.plotly_chart(fig, use_container_width=True)
    
    # Temperature analysis
    st.subheader("🌡️ Temperature Impact")
    
    data['temp_category'] = pd.cut(data['Temperature'], 
                                 bins=[0, 20, 25, 30, 35, 50], 
                                 labels=['Cold (<20°C)', 'Cool (20-25°C)', 'Moderate (25-30°C)', 
                                        'Warm (30-35°C)', 'Hot (>35°C)'])
    
    temp_impact = data.groupby('temp_category')['Time_taken(min)'].mean()
    
    fig = px.bar(x=temp_impact.index, y=temp_impact.values,
                title="Delivery Time by Temperature Range")
    st.plotly_chart(fig, use_container_width=True)
    
    # Weather and city interaction
    weather_city = data.groupby(['City', 'Weather_Condition'])['Time_taken(min)'].mean().reset_index()
    
    fig = px.bar(weather_city, x='City', y='Time_taken(min)', color='Weather_Condition',
                title="Weather Impact by City", barmode='group')
    st.plotly_chart(fig, use_container_width=True)

def show_dashboard_page(data):
    st.markdown('<h2 class="sub-header">📱 Real-time Dashboard</h2>', unsafe_allow_html=True)
    
    # Real-time metrics (simulated)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        current_orders = np.random.randint(1200, 1500)
        st.metric("🚀 Active Orders", f"{current_orders:,}")
    
    with col2:
        avg_time_today = data['Time_taken(min)'].mean() + np.random.normal(0, 2)
        st.metric("⏱️ Avg Time Today", f"{avg_time_today:.1f} min")
    
    with col3:
        delivery_partners = np.random.randint(800, 1000)
        st.metric("🚚 Active Partners", f"{delivery_partners}")
    
    with col4:
        satisfaction_today = data['Customer_Satisfaction'].mean() + np.random.normal(0, 0.1)
        st.metric("😊 Satisfaction Today", f"{satisfaction_today:.1f}/5")
    
    with col5:
        revenue_today = np.random.randint(500000, 800000)
        st.metric("💰 Revenue Today", f"₹{revenue_today/1000:.0f}K")
    
    with col6:
        delays = np.random.randint(50, 150)
        st.metric("⚠️ Delayed Orders", f"{delays}")
    
    # Live charts (simulated real-time data)
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly orders today
        current_hour = datetime.now().hour
        hours = list(range(max(0, current_hour - 6), current_hour + 1))
        orders = [np.random.randint(80, 150) for _ in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=orders, mode='lines+markers', name='Orders'))
        fig.update_layout(title="Orders in Last 6 Hours", xaxis_title="Hour", yaxis_title="Orders")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Current delivery time by city
        cities = data['City'].unique()[:6]
        current_times = [data[data['City'] == city]['Time_taken(min)'].mean() + np.random.normal(0, 3) for city in cities]
        
        fig = px.bar(x=cities, y=current_times, title="Current Avg Delivery Time by City")
        st.plotly_chart(fig, use_container_width=True)
    
    # Alert system
    st.subheader("🚨 Alert System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if np.random.random() > 0.7:
            st.markdown("""
            <div class="warning-card">
                <h4>⚠️ High Delivery Times Alert</h4>
                <p>Mumbai showing 15% higher than usual delivery times due to heavy traffic</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if np.random.random() > 0.8:
            st.markdown("""
            <div class="warning-card">
                <h4>🌧️ Weather Alert</h4>
                <p>Rain expected in Bangalore - expect 5-8 min delays</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if np.random.random() > 0.9:
            st.markdown("""
            <div class="success-card">
                <h4>🎉 Performance Alert</h4>
                <p>Delhi achieving record low delivery times today!</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Recent orders table
    st.subheader("📋 Recent Orders (Simulated)")
    recent_orders = data.sample(20)[['ID', 'City', 'Type_of_order', 'Time_taken(min)', 'Customer_Satisfaction', 'Type_of_vehicle']]
    recent_orders['Status'] = np.random.choice(['Delivered', 'In Transit', 'Preparing'], 20, p=[0.7, 0.2, 0.1])
    st.dataframe(recent_orders, use_container_width=True)

def show_business_page(data):
    st.markdown('<h2 class="sub-header">💰 Business Intelligence</h2>', unsafe_allow_html=True)
    
    # Revenue analysis
    total_revenue = data['Order_Value'].sum()
    avg_order_value = data['Order_Value'].mean()
    total_orders = len(data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Total Revenue", f"₹{total_revenue/1000000:.1f}M")
    
    with col2:
        st.metric("📊 Total Orders", f"{total_orders:,}")
    
    with col3:
        st.metric("💳 Avg Order Value", f"₹{avg_order_value:.0f}")
    
    with col4:
        customer_count = data['Customer_ID'].nunique()
        st.metric("👥 Unique Customers", f"{customer_count:,}")
    
    # Revenue by city
    city_revenue = data.groupby('City').agg({
        'Order_Value': ['sum', 'mean', 'count']
    }).round(0)
    city_revenue.columns = ['Total Revenue', 'Avg Order Value', 'Order Count']
    city_revenue['Revenue per Order'] = city_revenue['Total Revenue'] / city_revenue['Order Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("💰 Revenue by City")
        st.dataframe(city_revenue, use_container_width=True)
    
    with col2:
        fig = px.pie(values=city_revenue['Total Revenue'], names=city_revenue.index,
                    title="Revenue Distribution by City")
        st.plotly_chart(fig, use_container_width=True)
    
    # Time-based analysis
    st.subheader("📈 Temporal Business Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by hour
        hourly_revenue = data.groupby('Order_Time')['Order_Value'].sum()
        fig = px.bar(x=hourly_revenue.index, y=hourly_revenue.values,
                    title="Revenue by Hour of Day")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Orders by day of week
        daily_orders = data.groupby('Day_of_Week').size()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = px.bar(x=day_names, y=daily_orders.values,
                    title="Orders by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    
    # Profitability analysis
    st.subheader("📊 Profitability Analysis")
    
    # Simulate costs and profit margins
    data_copy = data.copy()
    data_copy['Delivery_Cost'] = (data_copy['distance'] * 8 +  # ₹8 per km
                                 data_copy['Time_taken(min)'] * 2 +  # ₹2 per minute
                                 np.random.normal(20, 5, len(data_copy)))  # Base cost
    
    data_copy['Commission'] = data_copy['Order_Value'] * 0.25  # 25% commission
    data_copy['Profit'] = data_copy['Commission'] - data_copy['Delivery_Cost']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_profit = data_copy['Profit'].sum()
        st.metric("💵 Total Profit", f"₹{total_profit/1000000:.1f}M")
    
    with col2:
        avg_profit_per_order = data_copy['Profit'].mean()
        st.metric("📈 Avg Profit/Order", f"₹{avg_profit_per_order:.0f}")
    
    with col3:
        profit_margin = (data_copy['Profit'].sum() / data_copy['Commission'].sum()) * 100
        st.metric("📊 Profit Margin", f"{profit_margin:.1f}%")
    
    # Profit by city
    city_profit = data_copy.groupby('City').agg({
        'Profit': ['sum', 'mean'],
        'Order_Value': 'sum'
    }).round(0)
    city_profit.columns = ['Total Profit', 'Avg Profit per Order', 'Revenue']
    city_profit['Profit Margin %'] = (city_profit['Total Profit'] / city_profit['Revenue'] * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("💰 Profitability by City")
        st.dataframe(city_profit, use_container_width=True)
    
    with col2:
        fig = px.bar(x=city_profit.index, y=city_profit['Profit Margin %'],
                    title="Profit Margin by City (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Cost analysis
    st.subheader("💸 Cost Analysis")
    
    vehicle_costs = data_copy.groupby('Type_of_vehicle').agg({
        'Delivery_Cost': 'mean',
        'Time_taken(min)': 'mean',
        'distance': 'mean'
    }).round(2)
    
    fig = px.bar(vehicle_costs, x=vehicle_costs.index, y='Delivery_Cost',
                title="Average Delivery Cost by Vehicle Type")
    st.plotly_chart(fig, use_container_width=True)
    
    # Key business insights
    st.subheader("🧠 Key Business Insights")
    
    insights = []
    
    # Find most profitable city
    most_profitable_city = city_profit['Total Profit'].idxmax()
    insights.append(f"🏆 Most profitable city: **{most_profitable_city}** with ₹{city_profit.loc[most_profitable_city, 'Total Profit']/1000:.0f}K profit")
    
    # Find best time for orders
    best_hour = data.groupby('Order_Time')['Order_Value'].sum().idxmax()
    insights.append(f"⏰ Peak revenue hour: **{best_hour}:00** - optimize partner allocation")
    
    # Vehicle efficiency
    most_efficient_vehicle = vehicle_costs['Delivery_Cost'].idxmin()
    insights.append(f"🚗 Most cost-efficient vehicle: **{most_efficient_vehicle}**")
    
    # Weekend performance
    weekend_revenue = data[data['Is_Weekend'] == 1]['Order_Value'].sum()
    weekday_revenue = data[data['Is_Weekend'] == 0]['Order_Value'].sum()
    weekend_orders = len(data[data['Is_Weekend'] == 1])
    weekday_orders = len(data[data['Is_Weekend'] == 0])
    
    if weekend_revenue/weekend_orders > weekday_revenue/weekday_orders:
        insights.append("📅 **Weekend orders** have higher average value - consider weekend promotions")
    else:
        insights.append("📅 **Weekday orders** are more valuable - focus on business lunch segment")
    
    for insight in insights:
        st.markdown(f"""
        <div class="insight-card">
            {insight}
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("💡 Strategic Recommendations")
    
    recommendations = [
        "🎯 **Focus on high-margin cities** - invest more delivery partners in profitable locations",
        "⏰ **Optimize peak hour pricing** - implement dynamic pricing during high-demand periods",
        "🚗 **Fleet optimization** - increase proportion of cost-efficient vehicles",
        "📊 **Customer segmentation** - develop loyalty programs for high-value customers",
        "🌟 **Partner incentives** - reward top-performing delivery partners to reduce churn",
        "📱 **Technology investment** - implement AI-powered route optimization",
        "🤝 **Restaurant partnerships** - negotiate better terms with high-volume restaurants"
    ]
    
    for rec in recommendations:
        st.markdown(f"""
        <div class="success-card">
            {rec}
        </div>
        """, unsafe_allow_html=True)

# Run the main application
if __name__ == "__main__":
    main()

# Additional helper functions for enhanced functionality

def generate_executive_summary(data):
    """Generate executive summary report"""
    summary = {
        'total_orders': len(data),
        'avg_delivery_time': data['Time_taken(min)'].mean(),
        'avg_satisfaction': data['Customer_Satisfaction'].mean(),
        'total_revenue': data['Order_Value'].sum(),
        'avg_distance': data['distance'].mean(),
        'peak_hour_impact': (data[data['Peak_Hour'] == 1]['Time_taken(min)'].mean() - 
                           data[data['Peak_Hour'] == 0]['Time_taken(min)'].mean()),
        'weather_impact': (data[data['Weather_Condition'] == 'Rain']['Time_taken(min)'].mean() - 
                          data[data['Weather_Condition'] == 'Clear']['Time_taken(min)'].mean()),
        'best_city': data.groupby('City')['Time_taken(min)'].mean().idxmin(),
        'worst_city': data.groupby('City')['Time_taken(min)'].mean().idxmax()
    }
    return summary

def calculate_delivery_efficiency_score(row):
    """Calculate efficiency score for delivery partners"""
    base_score = 100
    
    # Time penalty
    if row['Time_taken(min)'] > 30:
        base_score -= (row['Time_taken(min)'] - 30) * 2
    
    # Distance bonus/penalty
    expected_time = row['distance'] * 3 + 15  # Expected time formula
    if row['Time_taken(min)'] < expected_time:
        base_score += (expected_time - row['Time_taken(min)']) * 1.5
    
    # Weather adjustment
    weather_factors = {'Clear': 0, 'Cloudy': -2, 'Rain': -5, 'Storm': -10}
    base_score += weather_factors.get(row.get('Weather_Condition', 'Clear'), 0)
    
    # Rating bonus
    if row.get('Delivery_person_Ratings', 0) >= 4.5:
        base_score += 10
    
    return max(0, min(100, base_score))

def export_insights_report(data, filename="delivery_insights.txt"):
    """Export comprehensive insights report"""
    report = f"""
FOOD DELIVERY ANALYTICS REPORT
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
====================================================

EXECUTIVE SUMMARY
-----------------
• Total Orders Analyzed: {len(data):,}
• Average Delivery Time: {data['Time_taken(min)'].mean():.1f} minutes
• Customer Satisfaction: {data['Customer_Satisfaction'].mean():.1f}/5
• Total Revenue: ₹{data['Order_Value'].sum()/1000000:.1f}M
• Average Distance: {data['distance'].mean():.1f} km

PERFORMANCE METRICS
-------------------
• Fast Deliveries (≤25 min): {(data['Time_taken(min)'] <= 25).sum():,} ({(data['Time_taken(min)'] <= 25).sum()/len(data)*100:.1f}%)
• Slow Deliveries (>45 min): {(data['Time_taken(min)'] > 45).sum():,} ({(data['Time_taken(min)'] > 45).sum()/len(data)*100:.1f}%)
• High Satisfaction Orders (≥4.0): {(data['Customer_Satisfaction'] >= 4.0).sum():,} ({(data['Customer_Satisfaction'] >= 4.0).sum()/len(data)*100:.1f}%)

CITY PERFORMANCE
----------------
Best Performing City: {data.groupby('City')['Time_taken(min)'].mean().idxmin()}
Worst Performing City: {data.groupby('City')['Time_taken(min)'].mean().idxmax()}

WEATHER IMPACT
--------------
Clear Weather: {data[data['Weather_Condition'] == 'Clear']['Time_taken(min)'].mean():.1f} min avg
Rain Impact: +{data[data['Weather_Condition'] == 'Rain']['Time_taken(min)'].mean() - data[data['Weather_Condition'] == 'Clear']['Time_taken(min)'].mean():.1f} min
Storm Impact: +{data[data['Weather_Condition'] == 'Storm']['Time_taken(min)'].mean() - data[data['Weather_Condition'] == 'Clear']['Time_taken(min)'].mean():.1f} min

PEAK HOUR ANALYSIS
------------------
Peak Hour Delay: +{data[data['Peak_Hour'] == 1]['Time_taken(min)'].mean() - data[data['Peak_Hour'] == 0]['Time_taken(min)'].mean():.1f} minutes
Weekend Impact: +{data[data['Is_Weekend'] == 1]['Time_taken(min)'].mean() - data[data['Is_Weekend'] == 0]['Time_taken(min)'].mean():.1f} minutes

VEHICLE EFFICIENCY
------------------
"""
    
    vehicle_stats = data.groupby('Type_of_vehicle')['Time_taken(min)'].mean()
    for vehicle, time in vehicle_stats.items():
        report += f"• {vehicle.title()}: {time:.1f} min average\n"
    
    report += f"""

RECOMMENDATIONS
---------------
1. Focus optimization efforts on {data.groupby('City')['Time_taken(min)'].mean().idxmax()} (highest delivery times)
2. Implement weather-specific protocols for rain/storm conditions
3. Optimize partner allocation during peak hours (12-13, 19-21)
4. Consider vehicle type optimization based on route characteristics
5. Implement customer feedback system improvements for satisfaction scores below 4.0

END OF REPORT
====================================================
"""
    
    return report

# Data quality checks
def validate_data_quality(data):
    """Perform data quality checks"""
    issues = []
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.any():
        issues.append(f"Missing values found in columns: {missing_values[missing_values > 0].index.tolist()}")
    
    # Check for unrealistic delivery times
    extreme_times = data[(data['Time_taken(min)'] < 5) | (data['Time_taken(min)'] > 120)]
    if len(extreme_times) > 0:
        issues.append(f"Found {len(extreme_times)} orders with unrealistic delivery times (<5 min or >120 min)")
    
    # Check for unrealistic distances
    extreme_distances = data[(data['distance'] < 0.1) | (data['distance'] > 100)]
    if len(extreme_distances) > 0:
        issues.append(f"Found {len(extreme_distances)} orders with unrealistic distances (<0.1 km or >100 km)")
    
    # Check for rating consistency
    rating_issues = data[(data['Delivery_person_Ratings'] < 1) | (data['Delivery_person_Ratings'] > 5)]
    if len(rating_issues) > 0:
        issues.append(f"Found {len(rating_issues)} records with invalid ratings (outside 1-5 range)")
    
    return issues

# Performance optimization suggestions
def suggest_optimizations(data, predictor_results):
    """Generate optimization suggestions based on analysis"""
    suggestions = []
    
    # Analyze high-delay orders
    high_delay_orders = data[data['Time_taken(min)'] > data['Time_taken(min)'].quantile(0.9)]
    
    if len(high_delay_orders) > 0:
        # Find common factors in delayed orders
        delay_factors = high_delay_orders.groupby(['City', 'Weather_Condition', 'Peak_Hour']).size().reset_index(name='Count')
        top_delay_factor = delay_factors.loc[delay_factors['Count'].idxmax()]
        
        suggestions.append(f"High Priority: Address delays in {top_delay_factor['City']} during {top_delay_factor['Weather_Condition']} weather")
    
    # Partner performance suggestions
    if 'Random Forest' in predictor_results:
        partner_performance = data.groupby('Delivery_person_ID')['Time_taken(min)'].mean()
        underperforming_partners = partner_performance[partner_performance > partner_performance.quantile(0.8)]
        
        if len(underperforming_partners) > 0:
            suggestions.append(f"Training needed for {len(underperforming_partners)} underperforming delivery partners")
    
    # Route optimization
    city_efficiency = data.groupby('City').apply(lambda x: x['distance'].sum() / x['Time_taken(min)'].sum()).sort_values()
    least_efficient_city = city_efficiency.index[0]
    
    suggestions.append(f"Route Optimization: Focus on improving efficiency in {least_efficient_city}")
    
    # Fleet management
    vehicle_efficiency = data.groupby('Type_of_vehicle').apply(lambda x: x['distance'].sum() / x['Time_taken(min)'].sum())
    best_vehicle = vehicle_efficiency.idxmax()
    
    suggestions.append(f"Fleet Management: Consider increasing {best_vehicle} allocation for better efficiency")
    
    return suggestions