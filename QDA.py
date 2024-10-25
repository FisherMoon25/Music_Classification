import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# Label encoder
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

## QUADRATIC DISCRIMINANT ANALYSIS
qda = QuadraticDiscriminantAnalysis()

df = data_preprocessing('project_train.csv')

x = df.drop(columns=['Label'])
y = df['Label']

x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=0.2)

# Fit QDA on the selected features
qda.fit(x_train, y_train)

# Evaluate the model
y_pred_qda = qda.predict(x_eval)
conf_matrix_qda = confusion_matrix(y_eval, y_pred_qda)

print("Confusion matrix QDA after feature selection: \n", conf_matrix_qda)

kf = KFold(n_splits=5, shuffle=True, random_state=2)
qda_scores = cross_val_score(qda, x, y, cv=kf, scoring='accuracy').mean()

print(f'Mean CV Accuracy QDA after feature selection: {qda_scores * 100:.2f}%')

# Bagging QDA on selected features
bagging_model = BaggingClassifier(qda, n_estimators=100)
qda_bagging_cv = cross_val_score(bagging_model, x, y, cv=kf, scoring='accuracy').mean()

bagging_model.fit(x_train, y_train)
y_pred_bagging = bagging_model.predict(x_eval)
conf_matrix_bagging = confusion_matrix(y_eval, y_pred_bagging)

print("Confusion matrix bagging QDA after feature selection: \n", conf_matrix_bagging)
print(f'Mean CV Accuracy bagging QDA after feature selection: {qda_bagging_cv * 100:.2f}%')


# Custom cross-validation strategy
custom_cv = StratifiedKFold(n_splits=10)

# Define the parameter grid for QDA
param_grid = {
    'reg_param': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 10 values between 0 and 1 for regularization parameter
}

# Initialize lists to store accuracies and standard deviations
accuracies_test = [[] for _ in range(10)]
std_test = [0] * 10

# Variables to track the best parameters and the highest accuracy across all runs
best_overall_params = None
best_overall_accuracy = 0
best_overall_pca_features = 0
best_random_state = 0

# Draw distinct random states
np.random.seed(0)
random_states = np.random.randint(0, 1000, 10)

# Perform the grid search for each random state
for random_state in random_states:
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)

    # Loop over the number of features to reduce using PCA
    for nb_features in range(1, 11):
        # Perform PCA
        pca = PCA(n_components=nb_features)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Initialize the QDA model
        qda = QuadraticDiscriminantAnalysis()

        # Perform grid search using cross-validation
        grid_search = GridSearchCV(estimator=qda, param_grid=param_grid, cv=custom_cv, scoring='accuracy')
        grid_search.fit(X_train_pca, y_train)

        # Get the best model from grid search
        best_qda = grid_search.best_estimator_

        # Predict on the test set with the best model
        y_pred = best_qda.predict(X_test_pca)

        # Calculate accuracy for this configuration
        test_accuracy = np.mean(y_pred == y_test)

        # Store the test accuracy for each feature set
        accuracies_test[nb_features - 1].append(test_accuracy)

        # Check if this is the best accuracy we've seen so far
        if test_accuracy > best_overall_accuracy:
            best_overall_accuracy = test_accuracy
            best_overall_params = grid_search.best_params_
            best_overall_pca_features = nb_features
            best_random_state = random_state

# Calculate the mean and standard deviation for each number of features
mean_accuracies = [np.mean(acc) for acc in accuracies_test]
std_accuracies = [np.std(acc) for acc in accuracies_test]

# Final cross-validation on best model
best_pca = PCA(n_components=best_overall_pca_features)
X_best_pca = best_pca.fit_transform(x)
best_qda_model = QuadraticDiscriminantAnalysis(**best_overall_params)
best_cv_scores = cross_val_score(best_qda_model, X_best_pca, y, cv=custom_cv, scoring='accuracy')
best_cv_accuracy = best_cv_scores.mean()
best_cv_std = best_cv_scores.std()

# Plot accuracies with error bars
plt.errorbar(range(1, 11), mean_accuracies, yerr=std_accuracies, fmt='-o', capsize=5)
plt.xlabel('Number of features')
plt.ylabel('Test accuracy')
plt.title('Test accuracy vs Number of features with Standard Deviation (Grid Search QDA)')
plt.show()

# Print best model cross-validation results
print(f"Best overall QDA parameters: {best_overall_params}")
print(f"Best overall test accuracy: {best_overall_accuracy * 100:.2f}%")
print(f"Best PCA features: {best_overall_pca_features}")
print(f"Best random state: {best_random_state}")
print(f"Best CV accuracy (cross-validation on full data): {best_cv_accuracy * 100:.2f}% ± {best_cv_std * 100:.2f}%")
