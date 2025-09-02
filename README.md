# Detector de Emociones

Este proyecto fue desarrollado como parte de la asignatura de **Visión por Computador** en la Universidad Autonoma de Barcelona.  
Consiste en un modelo de **red neuronal convolucional (CNN)** capaz de detectar hasta **7 emociones** a partir de imágenes faciales:

- 😀 Contento  
- 😠 Enfadado  
- 😢 Triste  
- 😲 Sorprendido  
- 😐 Neutral  
- 😨 Temeroso  
- 🤢 Disgustado (este último con menor precisión)


## Tecnologías utilizadas

- Python 3  
- TensorFlow / Keras  
- OpenCV  
- NumPy 


## Ejecución

1. Clonar este repositorio:
   ```bash
   git clone https://github.com/1631153/Detector-emociones.git
2. Instalar dependencias:
    ```bash
    pip install -r requirements.txt
3. Entrenar o probar el modelo:
    ```bash
    python DeepLearning_train.py
    python DeepLearning_display.py

## Resultados

- **Accuracy alcanzado**: ~63%  
- Se realizaron varias versiones del modelo, pero esta fue la más sólida en cuanto a resultados.  
- Con más tiempo y recursos, podría optimizarse usando técnicas adicionales de **data augmentation**, arquitecturas más avanzadas o modelos preentrenados.

## Demo
https://github.com/user-attachments/assets/161c2560-cca1-433a-b6ae-1b562f37bce8
