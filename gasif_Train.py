import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.ensemble import  GradientBoostingRegressor


#-----------------
#Load dataset
#-----------------
df = pd. read_csv("gasification   ml.csv")

#droping data col
df = df.drop(columns=['#', 'Ref', 'Biomass species','Tar (g/m^3)','Gas (m3/kg)'], errors='ignore')

# Drop rows where target (H2) is missing
df = df.dropna(subset=['H2'])
print(df)


#-----------------
#Target and features
#-----------------

X = df.drop('H2',axis=1)
X = X.drop(columns=['CO', 'CO2', 'CH4'], errors='ignore')
y = df['H2']

#-----------------
#Num and Cat col
#-----------------
numeric_features = X.select_dtypes(include = ['int64','float64']).columns
categorical_features = X.select_dtypes(include = ['object']).columns

#-----------------
#Pipeline
#-----------------

#for numerical features
num_transformer = Pipeline (
    steps = [
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)
#for cat features
cat_transformer = Pipeline( steps = [
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
] )

#-----------------
#Preprocessing
#-----------------

preprocessor = ColumnTransformer(
    transformers= [
        ('num',num_transformer,numeric_features),
        ('cat',cat_transformer,categorical_features)
    ]
    )


#-----------------
#Gradiant boosting Reg. Model
#-----------------

reg_gb = GradientBoostingRegressor(learning_rate=0.01, 
                                    max_depth=20,
                                    n_estimators=300, 
                                    random_state=42,
                                    subsample=0.6)

#-----------------
#pipeline
#-----------------

gb_pipeline = Pipeline(
    [
        ('preprocessor',preprocessor),
        ('model',reg_gb)

     ]  )

#-----------------
#train-test split
#-----------------

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 , random_state=42)
gb_pipeline.fit(X_train,y_train)
y_pred = gb_pipeline.predict(X_test)
r2 = r2_score(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")\


#-----------------
#SAve model
#-----------------
with open("h2_gradient_boosting_model.pkl", "wb") as f:
    pickle.dump(gb_pipeline, f)

print("gb_pipeline svaed as h2_gradient_boosting_model.pkl")