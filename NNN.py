import tensorflow as tf
import keras
from keras import layers
from keras.utils import image_dataset_from_directory, load_img, img_to_array
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)



train_dataset = image_dataset_from_directory(
    'datasets/',
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary'
)

val_dataset = image_dataset_from_directory(
    'datasets/',
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary'
)

class_names = train_dataset.class_names
print("Class names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(AUTOTUNE)

model = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),         # height, width, channels

    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation='relu'),   # 32 filters, 3x3 kernel
    layers.MaxPooling2D((2, 2)),             # shrinks spatial size by half

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),                        # converts 3D → 1D for Dense layers
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')    # binary: cat vs dog
])


model.compile(
    optimizer='adam',               # Adaptive learning rate optimizer
    loss='binary_crossentropy',     # Standard loss for binary classification
    metrics=['accuracy'])

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=[early_stop]        
)

loss, accuracy = model.evaluate(val_dataset)
print(f"Validation accuracy: {accuracy:.2f}")



#Prediction
def predict_image(path):
    img = load_img(path, target_size=(64, 64))
    img_array = img_to_array(img)                 
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array, verbose=0)[0][0]

   
    label = class_names[1] if prob > 0.5 else class_names[0]
    confidence = prob if prob > 0.5 else 1 - prob
    print(f"{path}: {label} ({confidence:.0%})")

predict_image('straw.jpg')
predict_image('jahoda.webp')
predict_image('blue.webp')


model.save('berry_classifier.keras')