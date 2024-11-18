import pandas as pd
import numpy as np

def preprocess_data(df):
    df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])
    df['Month'] = df['Date Rptd'].dt.month
    df['Day of Week'] = df['Date Rptd'].dt.weekday + 1

    def categorize_crime(crime_desc):
        if 'THEFT' in crime_desc:
            return 'Theft'
        elif 'ASSAULT' in crime_desc:
            return 'Assault'
        elif 'BURGLARY' in crime_desc:
            return 'Burglary'
        else:
            return 'Other'

    df['Crime Category'] = df['Crm Cd Desc'].apply(categorize_crime)

    def severity_crime(crime_desc):
        if 'MURDER' in crime_desc or 'RAPE' in crime_desc or 'ROBBERY' in crime_desc:
            return 'High'
        elif 'ASSAULT' in crime_desc or 'BURGLARY' in crime_desc:
            return 'Medium'
        else:
            return 'Low'

    df['Crime Severity'] = df['Crm Cd Desc'].apply(severity_crime)

    df.rename(columns={
        'DR_NO': 'Report Number',
        'Date Rptd': 'Date Reported',
        'DATE OCC': 'Date Occurred',
        'TIME OCC': 'Time Occurred',
        'AREA': 'Area Code',
        'AREA NAME': 'Area Name',
        'Rpt Dist No': 'Report District Number',
        'Part 1-2': 'Crime Part',
        'Crm Cd': 'Crime Code',
        'Crm Cd Desc': 'Crime Description',
        'Mocodes': 'MO Codes',
        'Vict Age': 'Victim Age',
        'Vict Sex': 'Victim Sex',
        'Vict Descent': 'Victim Descent',
        'Premis Cd': 'Premises Code',
        'Premis Desc': 'Premises Description',
        'Weapon Used Cd': 'Weapon Used Code',
        'Weapon Desc': 'Weapon Description',
        'Status': 'Crime Status',
        'Status Desc': 'Status Description',
        'Crm Cd 1': 'Primary Crime Code',
        'LOCATION': 'Location',
        'Cross Street': 'Cross Street',
        'LAT': 'Latitude',
        'LON': 'Longitude'
    }, inplace=True)

    df['Crime Level'] = (df['Crime Code'] / 100).apply(np.floor)

    df = df[df['Victim Age'] > 0]
    age_bins = [0, 18, 35, 55, 100]
    age_labels = ['Child (0-18)', 'Young Adult (19-35)', 'Adult (36-55)', 'Senior (56+)']
    df = df.copy()
    df['Age Category'] = pd.cut(df['Victim Age'], bins=age_bins, labels=age_labels, right=False)



    df = df.dropna(subset=['Victim Sex'])
    sex_counts = df['Victim Sex'].value_counts()
    threshold = 10000
    sex_to_other = sex_counts[sex_counts < threshold].index
    df.loc[:, 'Victim Sex'] = df['Victim Sex'].replace(sex_to_other, 'Other')

    descent_counts = df['Victim Descent'].value_counts()
    threshold = 10000
    descent_to_other = descent_counts[descent_counts < threshold].index
    df.loc[:, 'Victim Descent'] = df['Victim Descent'].replace(descent_to_other, 'Other')

    df = df.dropna(subset=['Victim Descent', 'Premises Code'])

    columns_to_drop = [
        'Report Number', 
        'Time Occurred', 
        'Report District Number', 
        'MO Codes', 
        'Premises Description', 
        'Weapon Used Code', 
        'Weapon Description', 
        'Crime Status', 
        'Status Description', 
        'Location', 
        'Cross Street', 
    ]
    df = df.drop(columns=columns_to_drop)

    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numerical = df[numerical_columns]
    threshold = 2
    mean = np.mean(df_numerical, axis=0)
    std = np.std(df_numerical, axis=0)
    outliers = np.where(np.abs(df_numerical - mean) > threshold * std)
    df_cleaned_numerical = df_numerical[(np.abs(df_numerical - mean) <= threshold * std).all(axis=1)]
    df_cleaned = df.copy()
    df_cleaned.loc[:, numerical_columns] = df_cleaned_numerical
    df.update(df_cleaned)

    df_encoded = pd.get_dummies(df[['Crime Category', 'Crime Severity', 'Victim Sex']])
    df_combined = pd.concat([df_encoded, df[['Victim Age', 'Crime Level', 'Primary Crime Code', 'Area Code', 'Premises Code', 'Month', 'Day of Week', 'Crime Part', 'Crime Code', 'Latitude', 'Longitude']]], axis=1)

    return df, df_combined

def preprocess():
   
    df_cleaned = pd.read_pickle("Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/cleaned_data.pkl")
    df_preprocessed, df_combined = preprocess_data(df_cleaned)
    
    df_preprocessed.to_pickle("Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/preprocessed_data.pkl")
    df_combined.to_pickle("Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/final_data.pkl")
