import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory, load_img, img_to_array
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)



dataset = image_dataset_from_directory(
    'datasets/',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary'    # binary because you have exactly 2 classes
)

dataset = dataset.shuffle(buffer_size=1000, seed=42)

model = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),         # height, width, channels

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

# Split dataset — 80% train, 20% validation
dataset_size = dataset.cardinality().numpy()
train_size = int(dataset_size * 0.8)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

#Early stopping if accuracy doesn't improve
#callback = keras.callbacks.EarlyStopping(
#    patience=5,        # stops if no improvement for 5 epochs
#    restore_best_weights=True   # rolls back to the best epoch
#)

# Train
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset, 
)

loss, accuracy = model.evaluate(val_dataset)
print(f"Validation accuracy: {accuracy:.2f}")

img = load_img('straw.jpg', target_size=(64, 64))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0] > 0.5:
    print("Strawberry")
else:
    print("Blueberry")

img2 = load_img('jahoda.webp', target_size=(64, 64))
img_array2 = img_to_array(img2) / 255.0
img_array2 = np.expand_dims(img_array2, axis=0)

prediction2 = model.predict(img_array2)

if prediction2[0] > 0.5:
    print("Strawberry")
else:
    print("Blueberry")

img3 = load_img('blue.webp', target_size=(64, 64))
img_array3 = img_to_array(img3) / 255.0
img_array3 = np.expand_dims(img_array3, axis=0)

prediction3 = model.predict(img_array3)

if prediction3[0] > 0.5:
    print("Strawberry")
else:
    print("Blueberry")

model.save('berry_classifier.keras')
