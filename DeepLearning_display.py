import cv2
import numpy as np
import tensorflow as tf

# Crear el modelo
emotion_recognizer = tf.keras.models.Sequential()

emotion_recognizer.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_recognizer.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_recognizer.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
emotion_recognizer.add(tf.keras.layers.Dropout(0.25))

emotion_recognizer.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_recognizer.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
emotion_recognizer.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_recognizer.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
emotion_recognizer.add(tf.keras.layers.Dropout(0.25))

emotion_recognizer.add(tf.keras.layers.Flatten())
emotion_recognizer.add(tf.keras.layers.Dense(1024, activation='relu'))
emotion_recognizer.add(tf.keras.layers.Dropout(0.5))
emotion_recognizer.add(tf.keras.layers.Dense(7, activation='softmax'))

#Cargar detector de emociones
emotion_recognizer.load_weights('model.weights.h5')
emotionsList = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_resized = cv2.resize(roi_gray, (48, 48))
        cropped_img = np.expand_dims(np.expand_dims(roi_gray_resized, -1), 0)
        emotion_label = emotion_recognizer.predict(cropped_img)
        
        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        maxindex = int(np.argmax(emotion_label))
        cv2.putText(frame, emotionsList[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Detector de emociones', cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
