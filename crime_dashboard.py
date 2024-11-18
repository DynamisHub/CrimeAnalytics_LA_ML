import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import pickle
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import os



def forecast_area(area_name, crime_counts_pivot):
    df_area = crime_counts_pivot[[area_name]].reset_index()
    df_area.columns = ['ds', 'y']

    model = Prophet()
    model.fit(df_area)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast, model


def crime_dashboard():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    preprocessed_data_path = os.path.join(current_dir, "../pickle_files/preprocessed_data.pkl")
    df_combined_path = os.path.join(current_dir, "../pickle_files/df_combined.pkl")
    crime_counts_pivot_path = os.path.join(current_dir, "../pickle_files/crime_counts_pivot.pkl")

    df = pd.read_pickle(preprocessed_data_path)
    df_combined = pd.read_pickle(df_combined_path)
    crime_counts_pivot = pd.read_pickle(crime_counts_pivot_path)
    optimal_clusters = 6

    inertias_path = os.path.join(current_dir, "../pickle_files/inertias.pkl")
    with open(inertias_path, 'rb') as f:
        inertias = pickle.load(f)

    st.title('Crime Analysis Dashboard')


    st.header('Basic Statistics')
    st.write(df_combined.describe())

    st.header('Elbow Method For Optimal k')
    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, 10), inertias, 'bo-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Inertia')
    ax2.set_title('Elbow Method For Optimal k')
    ax2.grid(True)
    st.pyplot(fig2)


    st.header(f'K-means Clustering with {optimal_clusters} Clusters')
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df_combined, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax3)
    ax3.set_xlabel('Principal Component 1')
    ax3.set_ylabel('Principal Component 2')
    ax3.set_title(f'K-means Clustering with {optimal_clusters} Clusters')
    st.pyplot(fig3)


    st.header('Forecasting Crime Rates')
    selected_area = st.selectbox('Select Area', crime_counts_pivot.columns)

    if st.button('Generate Forecast'):
        forecast, model = forecast_area(selected_area, crime_counts_pivot)
        fig4 = model.plot(forecast)
        st.pyplot(fig4)

        st.subheader('Forecast Components')
        fig5 = model.plot_components(forecast)
        st.pyplot(fig5)


    st.header('Heatmap of Crimes by Area')
    fig6, ax6 = plt.subplots(figsize=(12, 8))
    sns.heatmap(crime_counts_pivot.T, cmap='viridis', ax=ax6)
    ax6.set_title('Heatmap of Crimes by Area')
    ax6.set_xlabel('Date Reported')
    ax6.set_ylabel('Area Name')
    st.pyplot(fig6)


    st.header('Crime Distribution by Month')
    fig7, ax7 = plt.subplots()
    sns.countplot(x='Month', data=df, ax=ax7, order=df['Month'].value_counts().index)
    ax7.set_title('Crime Distribution by Month')
    ax7.set_xlabel('Month')
    ax7.set_ylabel('Count')
    st.pyplot(fig7)

    st.header('Crime Distribution by Day of the Week')
    fig8, ax8 = plt.subplots()
    sns.countplot(x='Day of Week', data=df, ax=ax8, order=df['Day of Week'].value_counts().index)
    ax8.set_title('Crime Distribution by Day of the Week')
    ax8.set_xlabel('Day of the Week')
    ax8.set_ylabel('Count')
    st.pyplot(fig8)


    st.header('Difference Between Date Reported and Date Occurred (Time Series)')

    reported_freq = df['Date Reported'].value_counts().sort_index()
    occurred_freq = df['Date Occurred'].value_counts().sort_index()

    reported_freq_ma = reported_freq.rolling(window=7).mean()
    occurred_freq_ma = occurred_freq.rolling(window=7).mean()

    fig9, ax9 = plt.subplots(figsize=(10, 6))
    reported_freq_ma.plot(kind='line', label='Date Reported (Moving Average)', ax=ax9)
    occurred_freq_ma.plot(kind='line', label='Date Occurred (Moving Average)', ax=ax9)
    ax9.set_xlabel('Date')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Distribution of Date Reported and Date Occurred (Moving Average)')
    ax9.legend()
    ax9.grid(True)
    st.pyplot(fig9)


    st.header('Crime Map by Location')

    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])

    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    crime_map = folium.Map(location=map_center, zoom_start=12)

    heat_data = [[row['Latitude'], row['Longitude']] for index, row in df.iterrows()]
    HeatMap(heat_data).add_to(crime_map)

    folium_static(crime_map)

    st.sidebar.header('Model Information')
    st.sidebar.write("""
This dashboard uses Principal Component Analysis (PCA) for dimensionality reduction
and K-means clustering to identify patterns in the crime data. You can select different
areas to forecast crime rates using the Prophet model, and explore various visualizations
to understand the distribution and trends in the data.
""")

    st.sidebar.header('User Guide')
    st.sidebar.write("""
1. Use the dropdown menu to select an area for forecasting.
2. Click on 'Generate Forecast' to see the forecasted crime rates.
3. Explore the visualizations to gain insights into the crime data.
""")


if __name__ == '__main__':
    crime_dashboard()