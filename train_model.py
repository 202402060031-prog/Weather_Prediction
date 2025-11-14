
# Training script to retrain model (same preprocessing as used here).
# Run: python train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

df = pd.read_csv("data/seattle_weather_15k.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weather'] = df['weather'].fillna('unknown')

features = ['precipitation','temp_min','wind','month','day','weather']
X = df[features]
y = df['temp_max']

numeric_features = ['precipitation','temp_min','wind','month','day']
numeric_transformer = StandardScaler()
categorical_features = ['weather']
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "model/temp_max_model.pkl")
print("Model trained and saved to model/temp_max_model.pkl")
