import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
df = pd.read_csv('./building_energy_data.csv')

# Define numerical and categorical features
numerical_features = ['Building_Age', 'Number_of_Floors', 'Total_Area', 'Number_of_Windows']
categorical_features = ['Insulation_Type', 'Heating_System', 'Cooling_System']

# Add new features
df['Average_Window_Size'] = df['Total_Area'] / df['Number_of_Windows']
df['Total_Insulation_Area'] = df['Total_Area'] * 0.5
df['Heating_Cooling_System_Efficiency'] = np.random.uniform(low=0.8, high=1.0, size=len(df))

# Update numerical features list
numerical_features.extend(['Average_Window_Size', 'Total_Insulation_Area', 'Heating_Cooling_System_Efficiency'])

# Initialize the scaler and fit on numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Initialize OneHotEncoder for categorical variables with updated parameter
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
one_hot_encoded = encoder.fit_transform(df[categorical_features])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Concatenate all features
df_encoded = pd.concat([df[numerical_features], one_hot_df], axis=1)

# Define X and y
X = df_encoded
y = df['Energy_Consumption']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# Save the model and processing objects for use in the Flask API
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(encoder, 'encoder.joblib')
joblib.dump(model, 'model.joblib')
