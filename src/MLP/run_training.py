# run_training.py (esqueleto)
import pandas as pd
import numpy as np
from utils.preprocessing import build_preprocessor, fit_preprocessor, transform_with_preprocessor, save_preprocessor
from utils.label_utils import fit_label_encoder, save_label_encoder
from .tf_model import build_mlp, compile_model, train_model
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns



def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    print("Mostrando gráfica de pérdida...")
    plt.figure()
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss durante el entrenamiento')
    plt.legend()
    plt.grid(True)
    plt.savefig('src/test/training_loss.png')
    plt.close()

def plot_accuracy(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy durante el entrenamiento')
    plt.legend()
    plt.grid(True)
    plt.savefig('src/test/training_accuracy.png')
    plt.close()



def run_training(input_csv: str,):
    # 1) Cargar train.csv (previamente generado con split_save.py)
    train = pd.read_csv(input_csv)
    target_col = 'col17'

    # 2) separar X,y
    X_df = train.drop(columns=[target_col])
    y_sr = train[target_col]

    # 2.1) split train / validation
    X_train_df, X_val_df, y_train_sr, y_val_sr = train_test_split(
        X_df,
        y_sr,
        test_size=0.1,
        stratify=y_sr,
        random_state=42
    )

    # 3) construir & fit preprocessor
    preprocessor = build_preprocessor()
    preprocessor = fit_preprocessor(preprocessor, X_train_df)
    # opcional: guardar preprocessor
    save_preprocessor(preprocessor, 'src/test/preprocessor.joblib')

    # 4) transformar X_train
    X_train = transform_with_preprocessor(preprocessor, X_train_df)  # numpy array
    X_val   = transform_with_preprocessor(preprocessor, X_val_df)

    # 5) label encode y
    le, y_train = fit_label_encoder(y_train_sr)
    save_label_encoder(le, 'src/test/label_encoder.joblib')

    y_val = le.transform(y_val_sr)
    n_classes = len(le.classes_)

    # 6) construir y compilar modelo
    input_dim = X_train.shape[1]
    model = build_mlp(input_dim=input_dim, hidden_layers=(64,32), output_units=n_classes,
                    activation='relu', dropout_rate=0.15, l2_reg=5e-5)
    model = compile_model(model, learning_rate=1e-3)

    # 7) entrenar (solo usando train.csv)
    history = train_model(
        model, 
        X_train, # type: ignore
        y_train, # type: ignore
        validation=(X_val, y_val),
        batch_size=32, 
        epochs=100,
        model_path='src/test/mlp_model.keras', 
        patience=10
    )
    

    # 8) el checkpoint 'mlp_model' contiene el mejor modelo guardado
    # Guardar el modelo final si no usaste checkpoint
    model.save('src/test/mlp_model_final.keras')

    plot_loss(history)
    plot_accuracy(history)

    # ===============================
    # A3: Evaluación en VALIDACIÓN
    # ===============================

    y_val_pred_probs = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)

    cm = confusion_matrix(y_val, y_val_pred)
    report = classification_report(y_val, y_val_pred, target_names=le.classes_)

    print("Reporte de clasificación (VALIDACIÓN):")
    print(report)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=le.classes_, # type: ignore
                yticklabels=le.classes_, # type: ignore
                cmap='Blues')

    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.title("Matriz de confusión (Validación)")
    plt.tight_layout()
    plt.savefig("src/test/confusion_matrix_val.png")
    plt.close()
