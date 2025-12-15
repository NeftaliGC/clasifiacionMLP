from InquirerPy.resolver import prompt
from EDA.EDA import run_eda
from MLP.run_training import run_training
from MLP.run_test import run_test
from utils import preprocessing, label_utils, split_save

main_menu = [
    {
        "type": "list",
        "name": "main_menu",
        "message": "Utilice ▲▼ para seleccionar una acción:",
        "choices": [
            "📊 Análisis EDA",
            "🧠 Entrenamiento del modelo",
            "🔮 Predicción",
            "---",
            "🚪 Salir",
        ]
    }
]

prediction_menu = [
    {
        "type": "input",
        "name": "test_csv",
        "message": "Coloque el archivo csv en la ruta de este proyecto src/test/ \n posteriormente ingrese el nombre del archivo aquí incluyendo extensión \n (ejemplo: model_test.csv):",
        "default": "model_test.csv"
    }
]

if __name__ == "__main__":
    stop = False
    while not stop:
        print("\n======= Menú Principal =======")
        respuesta = prompt(main_menu)

        match respuesta["main_menu"]:
            case "📊 Análisis EDA":
                print("Realizando EDA")
                run_eda('src/data/input.csv', output_dir='src/EDA/results', target_col='col17')

                print("✅ Proceso finalizado correctamente.")
                input("Presiona Enter para volver al menú...")


            case "🧠 Entrenamiento del modelo":
                split_save.split_save_csv('src/data/input.csv', train_csv='src/data/model_train.csv', test_csv='src/test/model_test.csv',
                                            test_size=0.2, random_state=42, target_col='col17')
                run_training('src/data/model_train.csv')

                print("✅ Proceso finalizado correctamente.")
                input("Presiona Enter para volver al menú...")

            case "🔮 Predicción":
                pred_response = prompt(prediction_menu)
                print(f"Realizando predicción con el archivo: {pred_response['test_csv']}")
                run_test(f'src/test/{pred_response["test_csv"]}')

                print("✅ Proceso finalizado correctamente.")
                input("Presiona Enter para volver al menú...")
                
            case "🚪 Salir":
                print("Saliendo...")
                stop = True