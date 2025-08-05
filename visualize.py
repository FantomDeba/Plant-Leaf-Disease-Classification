import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

''' Add the files onto path '''
def create_data_set(path):
    images = []
    labels = []
    folders = os.listdir(path)
    for folder in folders:
        folder_path = os.path.join(path , folder)
        items = os.listdir(folder_path)
        for item in items:
            file_path = os.path.join(folder_path , item)
            images.append(file_path)
            labels.append(folder)
    df = pd.DataFrame({"file_paths" : images , "labels" : labels})
    return df
data = create_data_set("E:\placement\plantVillageDataClassification\PlantVillage")
import matplotlib.pyplot as plt
x_axis = data["labels"].value_counts().index
y_axis = data["labels"].value_counts().values
explode = []
nunique = data["labels"].nunique()
for i in range(nunique):
    explode.append(0.1)
# Set up a bigger figure size
plt.figure(figsize=(12, 12))

# Create pie chart
plt.pie(
    y_axis,
    labels=None,  # remove labels from slices
    explode=explode,
    autopct="%1.1f%%",
    startangle=140,
    textprops={'fontsize': 10}
)

# Add legend instead of slice labels
plt.legend(
    x_axis,
    title="Classes",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=10
)

# Clean and correct title
plt.title("Pie plot to see the distribution of the data", fontsize=14)

# Display
plt.tight_layout()
plt.show()
# Sort values for better visual flow (optional)
sorted_indices = y_axis.argsort()[::-1]  # descending order
x_sorted = [x_axis[i] for i in sorted_indices]
y_sorted = [y_axis[i] for i in sorted_indices]
plt.figure(figsize=(14, 8))

# Create bar plot
plt.barh(x_sorted, y_sorted, color='skyblue', edgecolor='black')
plt.gca().invert_yaxis()
plt.xlabel("Number of Images", fontsize=12)
plt.title("Bar plot showing the distribution of the data", fontsize=14)
for index, value in enumerate(y_sorted):
    plt.text(value + 1, index, str(value), va='center', fontsize=10)

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

# Create a reasonable figure size
plt.figure(figsize=(12, 12))

# Shuffle the dataset
shuffled_data, _ = train_test_split(data, test_size=0.2, random_state=45, shuffle=True)
for i, image in enumerate(shuffled_data["file_paths"][:25]):
    plt.subplot(5, 5, i + 1)
    img = cv2.imread(image)
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_cvt)
    plt.axis('off')  # remove axes ticks
    # Wrap long titles and reduce font size
    label = shuffled_data['labels'].iloc[i]
    plt.title(f"disease:\n{label}", fontsize=8)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

train_data , dummy_data = train_test_split(data , test_size = 0.3 , random_state = 44 , shuffle = True , stratify = data["labels"])
valid_data , test_data = train_test_split(dummy_data , test_size = 0.5 , random_state = 44 , shuffle = True , stratify = dummy_data["labels"]) 
print(test_data)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    x_col = "file_paths",
    y_col = "labels",
    target_size=(256, 256),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True
)
test_augmentated = test_datagen.flow_from_dataframe(test_data,
    x_col = "file_paths",
    y_col = "labels",
    target_size=(256, 256),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False)
images , labels = next(training_augmentation)
labels_index = np.argmax(labels , axis = 1)
dictionary = {v:k for k,v in training_augmentation.class_indices.items()}
aug_labels = [dictionary[i] for i in labels_index]
plt.figure(figsize = (10, 10))
for i ,image in enumerate(images): 
    plt.subplot(4,8,i+1)
    plt.imshow(image)
    
    clean_label = aug_labels[i].replace("Tomato_", "").replace("Pepper__bell___", "").replace("Potato_", "")
    clean_label = clean_label.replace("Septoria_leaf_spot", "Septoria Spot").replace("Spider_mites_Two-spotted_mite", "Spider Mites")
    plt.title(f"Aug: {clean_label}", fontsize=8) 
    plt.axis('off')
plt.tight_layout() 
plt.show()
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense , Dropout , BatchNormalization , Flatten , Activation
base_model = EfficientNetB3(include_top=False,
    weights='imagenet',
    input_shape=(256 , 256 , 3),
    pooling="max"
    )
for i,layer in enumerate(base_model.layers):
    if i in range (320,386):
        layer.trainable = True
    else:
        layer.trainable = False
len(base_model.layers)
from tensorflow.keras.layers import Dense , Dropout , BatchNormalization , Flatten,Conv2D , MaxPool2D , Input
from tensorflow.keras.models import Sequential
model = Sequential([
    Input(shape = (256 , 256 , 3)),
    Conv2D(32 , (3,3) , strides = (1,1) , padding = "same" , activation = "relu"),
    MaxPool2D(pool_size = (2,2) , strides = (2,2)),
    Conv2D(64 , (3,3) , strides = (1,1) , padding = "same" , activation = "relu"),
    MaxPool2D(pool_size = (2,2) , strides = (2,2)),
    Conv2D(128 , (3,3) , strides = (1,1) , padding = "same" , activation = "relu"),
    MaxPool2D(pool_size = (2,2) , strides = (2,2)),
    MaxPool2D(pool_size = (2,2) , strides = (2,2)),
    Flatten(),
    Dense(128 , activation = "relu"),
    Dropout(0.2),
    Dense(64 , activation = "relu"),
    BatchNormalization(),
    Dense(15 , activation = "softmax")
])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping
model.compile("adam" , loss = "categorical_crossentropy" , metrics = ["accuracy"])
early_stopping = EarlyStopping(
    monitor = "val_accuracy" ,
    patience = 4 ,
    restore_best_weights = True)
val_augmentated = validation_datagen.flow_from_dataframe(valid_data,
    x_col = "file_paths",
    y_col = "labels",
    target_size=(256, 256),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False)
history = model.fit(training_augmentation ,
                    batch_size = 32 ,
                    epochs = 5,
                    validation_data = val_augmentated,
                    callbacks = [early_stopping])
test_loss , test_accuracy = model.evaluate(test_augmentated)
print(test_loss)
print("-------------")
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("accuracy for training and validation")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(["training" , "validation"] , loc = "upper left")
plt.grid()
plt.show()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("loss for training and validation")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["training" , "validation"] , loc = "upper left")
plt.grid(True)
plt.show()
prediction = model.predict(test_augmentated)
from sklearn.metrics import classification_report
predictions = np.argmax(prediction , axis = 1)
test_labels = test_augmentated.classes
print(classification_report(test_labels , predictions))