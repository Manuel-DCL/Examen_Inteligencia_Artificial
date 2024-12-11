import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Desactivar ciertas optimizaciones de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Función para preprocesar imágenes
def preprocess_image(image, target_size=(128, 128)):
    if isinstance(image, str):  # Si es una ruta, cargar la imagen
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB
    image = cv2.resize(image, target_size)  # Redimensionar
    return image / 255.0  # Normalizar entre 0 y 1

# Función para cargar rutas de imágenes y etiquetas desde un directorio
def load_image_paths_and_labels(base_dir):
    image_paths = []
    labels = []
    class_names = os.listdir(base_dir)  # Obtener nombres de carpetas (clases)
    print(class_names)
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):  # Asegurar que es un directorio
            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filtrar imágenes
                    image_paths.append(file_path)
                    labels.append(label)
    return image_paths, labels, class_names

# Función para cargar y preprocesar las imágenes
def load_dataset(image_paths, labels, target_size=(128, 128)):
    images = [preprocess_image(img, target_size) for img in image_paths]
    return np.array(images), np.array(labels)

# Directorio base con subcarpetas por clase
base_dir = './media/'

# Cargar rutas de imágenes, etiquetas y nombres de clases
image_paths, labels, class_names = load_image_paths_and_labels(base_dir)

# Cargar y preprocesar las imágenes
X, y = load_dataset(image_paths, labels, target_size=(128, 128))
y = to_categorical(y, num_classes=len(class_names))  # Codificar etiquetas en formato one-hot

# Dividir datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo
model = Sequential([
    Flatten(input_shape=(128, 128, 3)),  # Aplana imágenes de entrada
    Dense(256, activation='relu'),  # Capa oculta con más neuronas
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')  # Capa de salida dinámica
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Guardar el modelo
model.save('plant_detector_model.h5')

# Convertir a TensorFlow.js
#!tensorflowjs_converter --input _format=keras ./plant_detector_model.h5 plant_detector_model.json

# Función para predecir la especie de la planta
def predict_species(image, model, target_size=(128, 128)):
    img = preprocess_image(image, target_size)
    img = np.expand_dims(img, axis=0)  # Expandir dimensiones
    predictions = model.predict(img)  # Realizar predicción
    class_idx = np.argmax(predictions)  # Índice de la clase predicha
    return class_idx

# Cargar el modelo guardado
model = tf.keras.models.load_model('plant_detector_model.h5')

# Procesar una imagen cargada para detección y predicción
image = cv2.imread('./muestra/muestra2.jpeg') # Ruta de la imagen para la detección y predicción
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
canny = cv2.Canny(gray, 10, 150)  # Detectar bordes
canny = cv2.dilate(canny, None, iterations=1)
canny = cv2.erode(canny, None, iterations=1)
cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterar sobre los contornos detectados
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)  # Obtener región de interés
    roi = image[y:y+h, x:x+w]  # Extraer región
    if roi.size > 0:
        class_idx = predict_species(roi, model)  # Predecir clase
        label = class_names[class_idx]  # Obtener etiqueta
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)  # Etiqueta en imagen
    cv2.drawContours(image, [c], 0, (0, 255, 0), 2)  # Dibujar contorno

# Mostrar la imagen con resultados
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Procesar un video desde la cámara para detección y predicción
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("No se pudo acceder a la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame de la cámara")
        break

    # Copiar el frame original para procesar
    image = frame.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    canny = cv2.Canny(gray, 10, 150)  # Detectar bordes
    canny = cv2.dilate(canny, None, iterations=1)  # Ampliar bordes
    canny = cv2.erode(canny, None, iterations=1)  # Reducir ruido
    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar sobre los contornos detectados
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)  # Obtener región de interés
        roi = image[y:y+h, x:x+w]  # Extraer la región de interés
        if roi.size > 0:
            # Asegurar tamaño correcto para el modelo
            roi = cv2.resize(roi, (128, 128))
            class_idx = predict_species(roi, model)  # Predecir clase
            label = class_names[class_idx]  # Obtener etiqueta de la clase
            # Dibujar la etiqueta y el contorno en la imagen
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.drawContours(image, [c], 0, (0, 255, 0), 2)

    # Mostrar la imagen procesada
    cv2.imshow('Detección en Video', image)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
