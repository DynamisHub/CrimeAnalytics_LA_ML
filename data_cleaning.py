import pandas as pd
from datetime import datetime

def clean_data(df):
    df['Date Rptd'] = pd.to_datetime(df['Date Rptd'].str.split().str[0], errors='coerce')
    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'].str.split().str[0], errors='coerce')

    df['LON'] = df['LON'].str.replace(r'\.(?=.*\.)', '', regex=True)
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')

    df['LAT'] = df['LAT'].apply(lambda x: x / 10 if abs(x) > 100 else x)
    df['LON'] = df['LON'].apply(lambda x: x / 10 if abs(x) > 200 else x)

    def convert_military_time_to_time(time_str):
        try:
            time_str = f"{time_str[:2]}:{time_str[2:]}"
            return datetime.strptime(time_str, '%H:%M').time()
        except ValueError:
            return None

    df['TIME OCC'] = df['TIME OCC'].apply(lambda x: convert_military_time_to_time(str(x).zfill(4)))

    columns_to_fill = ['Weapon Used Cd', 'Weapon Desc', 'Premis Desc', 'Cross Street', 'Mocodes']
    df[columns_to_fill] = df[columns_to_fill].fillna('Unknown')

    most_frequent_value = df['Crm Cd 1'].mode()[0]
    df['Crm Cd 1'] = df['Crm Cd 1'].fillna(most_frequent_value)

    df = df.drop(columns=['Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4'])

    duplicates = df.duplicated()
    num_duplicates = duplicates.sum()
    print(f"Number of duplicate rows: {num_duplicates}")

    return df

def clean():
    
    df = pd.read_pickle("Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/loaded_data.pkl")
    df_cleaned = clean_data(df)
    
    df_cleaned.to_pickle("Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/cleaned_data.pkl")

