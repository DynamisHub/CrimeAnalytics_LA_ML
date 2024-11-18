import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

def eda():
    df = pd.read_pickle("Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/preprocessed_data.pkl")
    df_combined = pd.read_pickle("Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/final_data.pkl")

    print(df.info())
    print(df.describe())

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Crime Severity', y='Victim Age', data=df)
    plt.title('Crime Severity vs Victim Age')
    plt.xlabel('Crime Severity')
    plt.ylabel('Victim Age')
    plt.savefig('Crime_Data_Analysis_in_LA_Using_ML_Techniques/eda_results/Crime_Severity_vs_Victim_Age.png')

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Crime Severity', hue='Victim Sex', data=df)
    plt.title('Crime Severity vs Victim Sex')
    plt.xlabel('Crime Severity')
    plt.ylabel('Count')
    plt.legend(title='Victim Sex', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('Crime_Data_Analysis_in_LA_Using_ML_Techniques/eda_results/Crime_Severity_vs_Victim_Sex.png')

    reported_freq = df['Date Reported'].value_counts().sort_index()
    occurred_freq = df['Date Occurred'].value_counts().sort_index()
    reported_freq_ma = reported_freq.rolling(window=7).mean()
    occurred_freq_ma = occurred_freq.rolling(window=7).mean()

    plt.figure(figsize=(10, 6))
    reported_freq_ma.plot(kind='line', label='Date Reported (Moving Average)')
    occurred_freq_ma.plot(kind='line', label='Date Occurred (Moving Average)')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.title('Distribution of Date Reported and Date Occurred (Moving Average)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Crime_Data_Analysis_in_LA_Using_ML_Techniques/eda_results/Distribution_of_Date_Reported_and_Date_Occurred.png')

    df['Date Difference'] = (df['Date Reported'] - df['Date Occurred']).dt.days
    
    plt.figure(figsize=(10, 6))
    plt.hist(df['Date Difference'], bins=range(-50, 51), edgecolor='k')
    plt.xlabel('Difference in Days')
    plt.ylabel('Frequency')
    plt.title('Difference Between Date Reported and Date Occurred')
    plt.grid(True)
    plt.savefig('Crime_Data_Analysis_in_LA_Using_ML_Techniques/eda_results/Difference_Between_Date_Reported_and_Date_Occurred.png')

    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])

    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    crime_map = folium.Map(location=map_center, zoom_start=12)

    heat_data = [[row['Latitude'], row['Longitude']] for index, row in df.iterrows()]
    HeatMap(heat_data).add_to(crime_map)

    plt.figure(figsize=(12, 6))
    sns.countplot(y='Area Name', data=df, order=df['Area Name'].value_counts().index)
    plt.title('Crime Distribution by Area')
    plt.xlabel('Count')
    plt.ylabel('Area')
    plt.savefig('Crime_Data_Analysis_in_LA_Using_ML_Techniques/eda_results/Crime_Distribution_by_Area.png')

    plt.figure(figsize=(12, 6))
    sns.histplot(df['Victim Age'], bins=30, kde=True)
    plt.title('Distribution of Victim Age')
    plt.xlabel('Victim Age')
    plt.ylabel('Frequency')
    plt.savefig('Crime_Data_Analysis_in_LA_Using_ML_Techniques/eda_results/Distribution_of_Victim_Age.png')

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Victim Sex', data=df, order=df['Victim Sex'].value_counts().index)
    plt.title('Crime Distribution by Victim Sex')
    plt.xlabel('Victim Sex')
    plt.ylabel('Count')
    plt.savefig('Crime_Data_Analysis_in_LA_Using_ML_Techniques/eda_results/Crime_Distribution_by_Victim_Sex.png')

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Month', data=df, order=df['Month'].value_counts().index)
    plt.title('Crime Distribution by Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.savefig('Crime_Data_Analysis_in_LA_Using_ML_Techniques/eda_results/Crime_Distribution_by_Month.png')

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Day of Week', data=df, order=df['Day of Week'].value_counts().index)
    plt.title('Crime Distribution by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Count')
    plt.savefig('Crime_Data_Analysis_in_LA_Using_ML_Techniques/eda_results/Crime_Distribution_by_Day_of_Week.png')

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Victim Descent', data=df, order=df['Victim Descent'].value_counts().index)
    plt.title('Crime Distribution by Victim Descent')
    plt.xlabel('Victim Descent')
    plt.ylabel('Count')
    plt.savefig('Crime_Data_Analysis_in_LA_Using_ML_Techniques/eda_results/Crime_Distribution_by_Victim_Descent.png')

    data_male = df[df['Victim Sex'] == 'M']['Victim Age'].dropna()
    data_female = df[df['Victim Sex'] == 'F']['Victim Age'].dropna()

    print(stats.shapiro(data_male))
    print(stats.shapiro(data_female))

    t_stat, p_value = stats.ttest_ind(data_male, data_female)

    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("We reject the null hypothesis (H0). There is a statistically significant difference between the mean ages of the victims.")
    else:
        print("We do not reject the null hypothesis (H0). There is no statistically significant difference between the mean ages of the victims.")


    model = ols('Q("Victim Age") ~ C(Q("Crime Category"))', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(anova_table)


