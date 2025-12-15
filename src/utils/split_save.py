# split_save.py
import pandas as pd
from sklearn.model_selection import train_test_split

def split_save_csv(input_csv: str, train_csv: str = 'model_train.csv', test_csv: str = 'model_test.csv',
                    test_size: float = 0.2, random_state: int = 42, target_col: str = 'col17'):
    """
    Divide el CSV en train/test y guarda ambos en disco SIN preprocesar.
    - input_csv: archivo original
    - train_csv/test_csv: rutas de salida
    - test_size: proporción para test
    """
    df = pd.read_csv(input_csv)
    train, test = train_test_split(df, test_size=test_size, stratify=df[target_col], random_state=random_state)
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)
    print(f"Guardado: {train_csv} ({len(train)} filas), {test_csv} ({len(test)} filas)")
    return train_csv, test_csv
