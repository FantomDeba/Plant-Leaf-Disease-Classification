import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
def create_data_set(path):
    images, labels = [], []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        for item in os.listdir(folder_path):
            file_path = os.path.join(folder_path, item)
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.abspath(file_path))
                labels.append(folder)
    return pd.DataFrame({"file_paths": images, "labels": labels})

def filter_invalid_images(df):
    valid_files, valid_labels = [], []
    for fp, label in zip(df["file_paths"], df["labels"]):
        try:
            img = cv2.imread(fp)
            if img is None or img.size == 0:
                continue
            valid_files.append(fp)
            valid_labels.append(label)
        except:
            continue
    return pd.DataFrame({"file_paths": valid_files, "labels": valid_labels})
DATASET_PATH = r"E:\placement\plantVillageDataClassification\PlantVillage\PlantVillage"

data = create_data_set(DATASET_PATH)
print("Original dataset size:", len(data))
data = filter_invalid_images(data)
print("After filtering invalid images:", len(data))

train_data, dummy_data = train_test_split(
    data, test_size=0.3, random_state=44, stratify=data["labels"]
)
valid_data, test_data = train_test_split(
    dummy_data, test_size=0.5, random_state=44, stratify=dummy_data["labels"]
)
print("Train:", len(train_data), " Val:", len(valid_data), " Test:", len(test_data))
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_augmentation = train_datagen.flow_from_dataframe(
    train_data,
    x_col="file_paths",
    y_col="labels",
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    shuffle=True
)

val_augmentated = validation_datagen.flow_from_dataframe(
    valid_data,
    x_col="file_paths",
    y_col="labels",
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

test_augmentated = test_datagen.flow_from_dataframe(
    test_data,
    x_col="file_paths",
    y_col="labels",
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)
num_classes = len(training_augmentation.class_indices)
print("Number of classes:", num_classes)
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [tf.shape(images)[0], -1, patch_dims])

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patch):
        num_patches = tf.shape(patch)[1]                
        positions = tf.range(start=0, limit=num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def create_vit_classifier(image_size=224, patch_size=16, projection_dim=64,
                          transformer_layers=6, num_heads=4, num_classes=15):
    num_patches = (image_size // patch_size) ** 2
    inputs = layers.Input(shape=(image_size, image_size, 3))
    patches = Patches(patch_size)(inputs)
    encoded = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
        x2 = layers.Add()([attn, encoded])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[projection_dim*2, projection_dim], dropout_rate=0.1)
        encoded = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=[512, 256], dropout_rate=0.5)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    return Model(inputs=inputs, outputs=outputs)

vit_model = create_vit_classifier(num_classes=num_classes)
vit_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

vit_model.summary()
train_labels = train_data["labels"].values
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("Class Weights:", class_weight_dict)
early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

history = vit_model.fit(
    training_augmentation,
    validation_data=val_augmentated,
    epochs=20,
    class_weight=class_weight_dict,   
    callbacks=[early_stopping]
)

# =========================================
# 6. Evaluation
# =========================================
test_loss, test_accuracy = vit_model.evaluate(test_augmentated)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.show()
y_pred = vit_model.predict(test_augmentated)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_augmentated.classes
print(classification_report(y_true, y_pred_classes, target_names=list(test_augmentated.class_indices.keys())))
