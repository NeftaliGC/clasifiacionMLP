# label_utils.py
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import numpy as np

def fit_label_encoder(y: pd.Series) -> Tuple[LabelEncoder, np.ndarray]:
    le = LabelEncoder()
    y_enc = np.asarray(le.fit_transform(y.astype(str)))
    return le, y_enc

def save_label_encoder(le, path: str):
    joblib.dump(le, path)

def load_label_encoder(path: str):
    return joblib.load(path)
