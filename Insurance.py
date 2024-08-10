# Import the needed packages
import pandas as pd
#import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
import pickle

#Dataset
data = pd.read_csv('Insurance_dataset.csv')

# Missing values in Dataset
missing_values = data.isnull().sum()

print("Missing Values:")
print(missing_values)

# Remove Duplicate values
data = data.drop_duplicates()
print("Data Duplicates:")
print(data)

# Encode the target 
label_encoder = LabelEncoder()
data['Outcome'] = label_encoder.fit_transform(data['Outcome'])

# Checking the target whether it is equally distributed
class_counts = data['Outcome'].value_counts()
class_proportions = data['Outcome'].value_counts(normalize=True)

print("Class Counts:")
print(class_counts)

print("\nClass Proportions:")
print(class_proportions)

#Droping target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

#Correlation matrix for numerical features
numerical_features = ['age', 'dur', 'num_calls']
correlation_matrix = X[numerical_features].corr()

#Correlation heatmap for numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

#Categorical features
target = 'Outcome'
categorical_features = ['job', 'marital', 'education_qual', 'call_type', 'day', 'mon', 'prev_outcome']

#Categorical features exist in the DataFrame
missing_features = [feature for feature in categorical_features if feature not in data.columns]
if missing_features:
    print("Missing features in the DataFrame:", missing_features)
else:
    data_for_corr = data.drop(columns=[target])

    #Convert categorical features to numerical format using one-hot encoding
    data_encoded = pd.get_dummies(data_for_corr[categorical_features])
    
    #Correlation matrix Categorical features
    correlation_matrix = data_encoded.corr()

    #Correlation heatmap Categorical features
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Categorical Features')
    plt.show()

# Outliers for numerical features
num_features = len(numerical_features)
plt.figure(figsize=(12, 4 * num_features))

# Nnumerical feature to create a box plot
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(num_features, 1, i)
    sns.boxplot(x=data[feature])
    plt.title(f'Box Plot for {feature}')
    plt.xlabel('') 

plt.tight_layout()
plt.show()

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#Numerical and categorical features
numerical_features = ['age', 'dur', 'num_calls']
categorical_features = ['job', 'marital', 'education_qual', 'call_type', 'day', 'mon', 'prev_outcome']

#Pipelines for numerical and categorical features
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#Combine pipelines for numerical and categorical features
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

#Preprocessor on the training data
preprocessor.fit(X_train)

#Apply Preprocessing to the training data
X_train_preprocessed = preprocessor.transform(X_train)

# Apply preprocessing to the validation data
X_val_preprocessed = preprocessor.transform(X_val)

# Apply preprocessing to the test data
X_test_preprocessed = preprocessor.transform(X_test)

#imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

#Models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train_resampled, y_train_resampled)
    
    # Feature importance
    if model_name != 'Logistic Regression':
        feature_names = numerical_features + list(preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_features))
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances * 100})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        print("Feature Importances:")
        print(importance_df)
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Feature Importances - {model_name}')
        plt.show()
    
    #Predictions on the validation set
    y_val_pred = model.predict(X_val_preprocessed)
    y_val_pred_prob = model.predict_proba(X_val_preprocessed)[:, 1]
    
    #Evaluate the model on the validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)*100
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_pred_prob)
    val_conf_matrix = confusion_matrix(y_val, y_val_pred)
    
    print(f'Validation Accuracy: {val_accuracy}')
    print(f'Validation Precision: {val_precision}')
    print(f'Validation Recall: {val_recall}')
    print(f'Validation F1-Score: {val_f1}')
    print(f'Validation ROC-AUC: {val_roc_auc}')
    print('Validation Confusion Matrix:')
    print(val_conf_matrix)
    
    # Save the model in pickle
    with open(f'{model_name.lower().replace(" ", "_")}_model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Make predictions on the testing data with the best model
best_model = models['Gradient Boosting']
y_test_pred = best_model.predict(X_test_preprocessed)
y_test_pred_prob = best_model.predict_proba(X_test_preprocessed)[:, 1]

# Evaluate the model on the testing set
test_accuracy = accuracy_score(y_test, y_test_pred)*100
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_pred_prob)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

print(f'Test Accuracy: {test_accuracy}')
print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')
print(f'Test F1-Score: {test_f1}')
print(f'Test ROC-AUC: {test_roc_auc}')
print('Test Confusion Matrix:')
print(test_conf_matrix)

#Customer separation using KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
X_for_clustering = preprocessor.transform(X)
clusters = kmeans.fit_predict(X_for_clustering)

#cluster labels to the dataset
data['Cluster'] = clusters

# Aggregate numeric features by cluster
numeric_data = data.select_dtypes(include=[np.number])
cluster_means = numeric_data.groupby('Cluster').mean()

print("Customer Segmentation based on Clusters:")
print(cluster_means)

# Plot of cluster distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Cluster', data=data)
plt.title('Customer Segmentation Clusters')

# Design
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), 
                textcoords='offset points')
plt.show()


# # Streamlit Title
# st.title("Insurance Subscription Prediction")

# #Input fields for the features
# st.header("Enter Customer Details:")

# #Numerical Input
# age = st.number_input("Age", min_value=18, max_value=100, value=30)
# dur = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=100)
# num_calls = st.number_input("Number of Calls", min_value=0, max_value=100, value=1)

# #Categorical Input
# job = st.selectbox("Job", ['admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
# marital = st.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'])
# education_qual = st.selectbox("Education Qualification", ['primary', 'secondary', 'tertiary', 'unknown'])
# call_type = st.selectbox("Call Type", ['cellular', 'telephone', 'unknown'])
# day = st.number_input("Day of the Month", min_value=1, max_value=31, value=1)
# mon = st.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
# prev_outcome = st.selectbox("Previous Outcome", ['failure', 'other', 'success', 'unknown'])

# # Dictionary from the inputs
# input_data = {
#     'age': [age],
#     'dur': [dur],
#     'num_calls': [num_calls],
#     'job': [job],
#     'marital': [marital],
#     'education_qual': [education_qual],
#     'call_type': [call_type],
#     'day': [day],
#     'mon': [mon],
#     'prev_outcome': [prev_outcome]
# }

# # Convert the dictionary to dataframe
# input_df = pd.DataFrame(input_data)

# # Preprocess the input data
# input_preprocessed = preprocessor.transform(input_df)

# # Predict the outcome using the pre-trained model
# prediction = model.predict(input_preprocessed)[0]

# # Prediction result
# if st.button("Predict"):
#     if prediction == 1:
#         st.success("The customer is likely to subscribe to the insurance policy.")
#     else:
#         st.warning("The customer is not likely to subscribe to the insurance policy.")
