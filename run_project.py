from data_loader import load
from data_cleaning import clean
from data_preprocessing import preprocess
from eda_analysis import eda
from ml_analysis import ml_analysis

# XXX: Some paths are relative to current and looking for Crime_Data_Analysis_in_LA_Using_ML_Techniques
# Hence I need to run the code from the root directory: 
# python Crime_Data_Analysis_in_LA_Using_ML_Techniques\scripts\run_project.py

def main():
    print('Running project...')
    print("loading")
    load()
    print("Data loaded successfully.")
    
    print("cleaning")    
    clean()
    print("Data cleaned successfully.")
    
    print("preprocessing")
    preprocess()
    print("Data preprocessed successfully.")
    
    print("eda")
    eda()
    print("EDA completed successfully.")
    
    print("ml analysis")
    ml_analysis()
    print("ML analysis completed successfully.")

if __name__ == '__main__':
    main()



## For running Dashboards : 
# .\Crime_Data_Analysis_in_LA_Using_ML_Techniques\scripts> python -m streamlit run crime_dashboard.py
# .\Crime_Data_Analysis_in_LA_Using_ML_Techniques\scripts> python -m streamlit run propability_dashboard.py
  