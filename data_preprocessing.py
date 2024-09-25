# Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler

# To use the module:
# import data_preprocessing as dp and call dp.data_preprocessing(file_name)


def data_preprocessing(file_name: str) -> pd.DataFrame:
    """
    This function reads the data set from the specified file, 
    checks the values in the specified columns are within the correct range, 
    standardizes the data, and returns the processed data set.
    Parameters:
    file_name (str): The name of the file containing the data set.

    Returns:
    pd.DataFrame: The processed data set.
    """
    # Read the data set
    df = pd.read_csv(file_name)

    # Ensure the values in the specified columns are between 0 and 1
    # and remove any rows with incorrect values
    columns_to_check = ['danceability', 'energy', 'speechiness', 'acousticness',
                        'instrumentalness', 'liveness', 'valence']
    df = df[(df[columns_to_check] >= 0).all(axis=1) &
            (df[columns_to_check] <= 1).all(axis=1)]

    # Check loudness values are within [-60, 0]
    # and remove any rows with incorrect values
    df = df[(df['loudness'] >= -60) & (df['loudness'] <= 0)]

    # Convert 'mode' values equal to 0 to -1
    df['mode'] = df['mode'].replace(0, -1)

    # TO DO - One hot encode the 'key' column?

    # Standardize the data, expect for categorical columns
    scaler = StandardScaler()
    columns_to_standardize = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                              'instrumentalness', 'liveness', 'valence', 'tempo']
    df[columns_to_standardize] = scaler.fit_transform(
        df[columns_to_standardize])

    return df
