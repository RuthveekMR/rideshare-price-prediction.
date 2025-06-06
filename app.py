import streamlit as st
import pandas as pd
import numpy as np
import joblib
import MinMaxScaler

# Drive direct download link
url = "https://drive.google.com/uc?id=1ChqFR9EjX8ajos805cPR6dB4haBN_cTn"

# Read CSV from Google Drive
df = pd.read_csv(url)

# Extract unique values
unique_cab_types = df['name'].unique().tolist()
unique_sources = df['source'].unique().tolist()
unique_destinations = df['destination'].unique().tolist()

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['time_stamp'] / 1000, unit='s')

# Extract useful time-based features
df['hour'] = df['timestamp'].dt.hour

# Scale distance
distance_df = df[['source', 'destination', 'distance']].drop_duplicates()
scaler = joblib.load("scaler.joblib")  # Ensure the MinMaxScaler is saved & loaded
distance_df['scaled_distance'] = scaler.transform(distance_df[['distance']])

# Function to get scaled distance
def get_scaled_distance(source, destination):
    row = distance_df[(distance_df['source'] == source) & (distance_df['destination'] == destination)]
    return row['scaled_distance'].values[0] if not row.empty else 0  # Default to 0 if not found

# Extract surge multiplier mapping
surge_df = (
    df.groupby(['source', 'destination', 'hour'])['surge_multiplier']
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

# Function to get surge multiplier
def get_surge_multiplier(source, destination, hour):
    row = surge_df[(surge_df['source'] == source) & (surge_df['destination'] == destination) & (surge_df['hour'] == hour)]
    return row['surge_multiplier'].values[0] if not row.empty else 1  # Default to 1

# Streamlit UI
st.title("🚖 Cab Fare Prediction (Uber & Lyft)")

# Sidebar input options
cab_type = st.sidebar.selectbox("Select Cab Type", unique_cab_types)
source = st.sidebar.selectbox("Select Source", unique_sources)
destination = st.sidebar.selectbox("Select Destination", unique_destinations)
hour = st.sidebar.slider("Select Hour of the Day", 0, 23, 12)

if st.sidebar.button("Predict Fare"):
    # Compute features
    scaled_distance = get_scaled_distance(source, destination)
    surge_multiplier = get_surge_multiplier(source, destination, hour)

    # Prepare input data
    input_data = pd.DataFrame(np.zeros((1, len(model.feature_names_))), columns=model.feature_names_)
    input_data['distance'] = scaled_distance
    input_data['surge_multiplier'] = surge_multiplier
    input_data['hour'] = hour

    # Set categorical flags correctly (fixing cab type encoding)
    if f"name_{cab_type}" in input_data.columns:  # ✅ Correcting cab type encoding
        input_data[f"name_{cab_type}"] = 1
    if f"source_{source}" in input_data.columns:
        input_data[f"source_{source}"] = 1
    if f"destination_{destination}" in input_data.columns:
        input_data[f"destination_{destination}"] = 1

    # Ensure input_data matches model feature names
    input_data = input_data[model.feature_names_]

    # Predict fare
    predicted_fare = model.predict(input_data)[0]

    # Show result
    st.success(f"Estimated Fare: **${round(predicted_fare, 2)}**")
