import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    print(f"Loaded {len(images)} images")
    print(f"Number of categories: {np.max(labels) + 1}")

    # Define model parameters
    IMG_WIDTH = 30  # Make sure these match your resize dimensions
    IMG_HEIGHT = 30
    NUM_CATEGORIES = 10
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
    
    if len(images) == 0:
        print("ERROR: No images loaded - check your data directory structure")
        sys.exit(1)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Define input shape based on your image dimensions
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)  # 3 channels for RGB
    
    # Pass input_shape to get_model
    model = get_model(input_shape=input_shape, num_categories=NUM_CATEGORIES)

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        best_model = sys.argv[2]
        model.save("best_model.h5")
        
    # Verify file was created
        if os.path.exists('best_model.h5'):
            print(f"Model successfully saved to: {os.path.abspath('best_model.h5')}")
            print(f"File size: {os.path.getsize('best_model.h5')/1024:.2f} KB")
        else:
            print("Error: File was not created!")
            

def load_data(data_dir, img_width=30, img_height=30):
    images = []
    labels = []
    
    try:
        categories = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
        categories.sort(key=int)
        
        if not categories:
            raise ValueError(f"No valid category folders found in {data_dir}")
        
        for category in categories:
            category_dir = os.path.join(data_dir, category)
            for img_file in os.listdir(category_dir):
                img_path = os.path.join(category_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, (img_width, img_height))
                    images.append(img)
                    labels.append(int(category))
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
                    
        if len(images) == 0:
            raise ValueError("No valid images found in any category folder")
            
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        sys.exit(1)
        
    return np.array(images), np.array(labels)
    raise NotImplementedError


def get_model(input_shape, num_categories):

    model = Sequential()
    
    # Convolutional Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', 
                   input_shape=input_shape,
                   kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    # Convolutional Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    
    # Convolutional Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # Output Layer
    model.add(Dense(num_categories, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
    raise NotImplementedError


if __name__ == "__main__":
    main()
