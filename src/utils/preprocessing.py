# preprocessing.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PowerTransformer
import joblib
import pandas as pd
from pathlib import Path
from typing import List, Tuple

# Columnas (ajusta si tus nombres son otros)
NUMERIC_COLS = ['col2','col3','col4','col7','col8','col11','col13','col14']
CATEGORICAL_COLS = ['col1','col5','col6','col9','col10','col12','col15','col16']

def build_preprocessor(numeric_cols: List[str] = NUMERIC_COLS,
                        categorical_cols: List[str] = CATEGORICAL_COLS,
                        use_robust_for_outliers: bool = False) -> ColumnTransformer:
    """
    Construye el ColumnTransformer preprocesador (sin fit).
    - PowerTransformer (Yeo-Johnson) -> StandardScaler para numéricas por defecto.
    - OneHotEncoder(handle_unknown='ignore') para categóricas.
    """
    numeric_steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', RobustScaler() if use_robust_for_outliers else StandardScaler())
    ]
    num_pipeline = Pipeline(numeric_steps)

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='drop')

    return preprocessor

def fit_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessor on X_train and return fitted transformer."""
    preprocessor.fit(X_train)
    return preprocessor

def transform_with_preprocessor(preprocessor: ColumnTransformer, X: pd.DataFrame):
    """Transform a dataframe using a fitted preprocessor (returns numpy array)."""
    return preprocessor.transform(X)

def save_preprocessor(preprocessor: ColumnTransformer, path: str):
    joblib.dump(preprocessor, path)

def load_preprocessor(path: str) -> ColumnTransformer:
    return joblib.load(path)
