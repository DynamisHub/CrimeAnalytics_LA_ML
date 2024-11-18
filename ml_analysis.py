import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def load_data():
    df = pd.read_pickle("Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/preprocessed_data.pkl")
    df_combined = pd.read_pickle("Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/final_data.pkl")
    return df, df_combined

def ml_analysis():
    df, df_combined = load_data()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_combined)

    pca = PCA(n_components=4)
    pca_data = pca.fit_transform(scaled_data)
    df_pca = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(4)])
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance.cumsum()

    inertias = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_pca)
        inertias.append(kmeans.inertia_)
    
    with open('Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/inertias.pkl', 'wb') as f:
        pickle.dump(inertias, f)

    optimal_clusters = 6
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df_pca['Cluster'] = kmeans.fit_predict(df_pca)

    df_pca['Victim Sex'] = df['Victim Sex'].values

    X = df_pca.drop(columns=['Victim Sex'])
    y_cluster = df_pca['Cluster']
    y_sex = df_pca['Victim Sex']

    X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(X, y_cluster, test_size=0.3, random_state=42)
    X_train_sex, X_test_sex, y_train_sex, y_test_sex = train_test_split(X, y_sex, test_size=0.3, random_state=42)

    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42),
    }

    for name, clf in classifiers.items():
        clf.fit(X_train_cluster, y_train_cluster)
        predictions = clf.predict(X_test_cluster)
        print(f"Classification report for Cluster with {name}:")
        print(classification_report(y_test_cluster, predictions))

    for name, clf in classifiers.items():
        clf.fit(X_train_sex, y_train_sex)
        predictions = clf.predict(X_test_sex)
        print(f"Classification report for Victim Sex with {name}:")
        print(classification_report(y_test_sex, predictions))

    df_forecasting = df[['Date Reported', 'Area Name']].copy()
    crime_counts = df_forecasting.groupby(['Date Reported', 'Area Name']).size().reset_index(name='Crime Count')
    crime_counts_pivot = crime_counts.pivot(index='Date Reported', columns='Area Name', values='Crime Count').fillna(0)
    crime_counts_pivot = crime_counts_pivot.asfreq('D', fill_value=0)

    df_combined = df_pca
    df_combined.to_pickle('Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/df_combined.pkl')
    crime_counts_pivot.to_pickle('Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/crime_counts_pivot.pkl')

    with open("Crime_Data_Analysis_in_LA_Using_ML_Techniques/pickle_files/random_forest_model.pkl", "wb") as file:
        pickle.dump(classifiers["Logistic Regression"], file)

    print("ML analysis completed and data saved to pickle files.")
