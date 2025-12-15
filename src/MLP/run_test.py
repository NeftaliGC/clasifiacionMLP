# run_test.py
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model

from utils.preprocessing import transform_with_preprocessor


def run_test(test_csv: str):
    target_col = 'col17'

    # 1) cargar test
    test = pd.read_csv(test_csv)
    X_test_df = test.drop(columns=[target_col])
    y_test_sr = test[target_col]

    # 2) cargar preprocessor
    preprocessor = joblib.load('src/test/preprocessor.joblib')

    # 3) transformar X_test
    X_test = transform_with_preprocessor(preprocessor, X_test_df)

    # 4) cargar label encoder
    le = joblib.load('src/test/label_encoder.joblib')
    y_test = le.transform(y_test_sr)

    # 5) cargar modelo
    model = load_model('src/test/mlp_model.keras')

    # 6) predicciones
    y_test_pred_probs = model.predict(X_test) # type: ignore
    y_test_pred = np.argmax(y_test_pred_probs, axis=1)

    # 7) métricas
    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(
        y_test,
        y_test_pred,
        target_names=le.classes_
    )

    print("Reporte de clasificación (TEST):")
    print(report)

    # 8) matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )

    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.title("Matriz de confusión (TEST)")
    plt.tight_layout()
    plt.savefig("src/test/confusion_matrix_test.png")
    plt.close()
