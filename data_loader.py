
import pandas as pd

def load_data(file_path):
     df = pd.read_csv(file_path, encoding="latin1", delimiter=';')
     return df
def load():
     file_path = './Crime_Data_Analysis_in_LA_Using_ML_Techniques/data/raw_data/data/Crime_Data_from_2020.csv'
     df = load_data(file_path)
     df.to_pickle("Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/loaded_data.pkl")

