import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

label_encoder = LabelEncoder()

def data_preprocessing(file_name: str) -> pd.DataFrame:
    """
    This function reads the data set from the specified file,
    checks the values in the specified columns are within the correct range,
    standardizes the data, and returns the processed data set.
    """
    df = pd.read_csv(file_name)

    # Ensure the values in the specified columns are between 0 and 1
    columns_to_check = ['danceability', 'energy', 'speechiness', 'acousticness',
                        'instrumentalness', 'liveness', 'valence']
    df = df[(df[columns_to_check] >= 0).all(axis=1) &
            (df[columns_to_check] <= 1).all(axis=1)]

    # Check loudness values are within [-60, 0]
    df = df[(df['loudness'] >= -60) & (df['loudness'] <= 0)]

    # Encode 'key' and 'mode' columns
    df['key'] = label_encoder.fit_transform(df['key'])
    df['mode'] = label_encoder.fit_transform(df['mode'].replace(0, -1))

    # Standardize the numerical columns
    scaler = StandardScaler()
    columns_to_standardize = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                              'instrumentalness', 'liveness', 'valence', 'tempo']
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

    return df

# Load and preprocess the data
df = data_preprocessing('project_train.csv')
x = df.drop(columns=['Label'])
y = df['Label']

# Get the distribution of the classes in the 'Label' column
#class_distribution = df['Label'].value_counts()

# Print the class distribution
#print(class_distribution,"\n")

# Split the data into training and evaluation sets
x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=0.2)

# Initialize logistic regression model
logistic_model = LogisticRegression(class_weight='balanced')

# Perform grid search to find the best hyperparameters
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'penalty': ['l2'],
    'max_iter': [100, 200, 500]
}

grid_search = GridSearchCV(logistic_model, param_grid, cv=10, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Best model from grid search
print("Best Parameters:\n", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluate the best model on the evaluation set
y_pred_grid = best_model.predict(x_eval)
accuracy_grid = accuracy_score(y_eval, y_pred_grid)
print("Accuracy with GridSearch: {:.2f}%".format(accuracy_grid * 100))

# Perform cross-validation on the entire dataset
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mean_accuracy = cross_val_score(best_model, x, y, cv=kf, scoring='accuracy').mean()
print(f'Mean CV Accuracy for the best grid search model: {mean_accuracy * 100:.2f}%')

# Apply PCA and evaluate the model again
pca = PCA(n_components=9)
x_pca_train = pca.fit_transform(x_train)
x_pca_eval = pca.transform(x_eval)
x_pca = pca.fit_transform(x)

# Train the model on PCA-transformed data and evaluate
best_model.fit(x_pca_train, y_train)
y_pred_pca = best_model.predict(x_pca_eval)
pca_mean_accuracy = cross_val_score(best_model, x_pca, y, cv=kf, scoring='accuracy').mean()
print(f'Mean CV Accuracy after PCA: {pca_mean_accuracy * 100:.2f}%')

conf_matrix_grid = confusion_matrix(y_eval, y_pred_grid)
conf_matrix_gridpca = confusion_matrix(y_eval, y_pred_pca)

print("Confusion matrix for best grid search model: \n", conf_matrix_grid)
print("Confusion matrix for best grid search model after PCA: \n", conf_matrix_gridpca)


## Final evaluation
"""
y_pred = logistic_model.predict(x_eval)
accuracy = accuracy_score(y_eval, y_pred)
conf_matrix = confusion_matrix(y_eval, y_pred)
class_report = classification_report(y_eval, y_pred)

print("Confusion matrix \n",conf_matrix)
print("Accuracy \n", round(accuracy*100, 2), "%")

scores = cross_val_score(logistic_model, x, y, cv=10, scoring='accuracy')
mean_accuracy = scores.mean()
print(f'Mean CV Accuracy logistic regression model: {mean_accuracy * 100:.2f}%')
"""


