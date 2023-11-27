import string
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

epochs = 10       # Number of epochs for training
img_rows, img_cols = None, None  # Will be determined based on the loaded image
digits_in_img = 6 # CAPTCHA length is now 6 characters
num_classes = 36  # 26 English alphabets + 10 digits
x_list, y_list = [], []
x_train, y_train, x_test, y_test = [], [], [], []

# Updated mapping for alphanumeric characters
characters = string.digits + string.ascii_uppercase  # Digits + uppercase letters
char_to_int = {char: idx for idx, char in enumerate(characters)}

# Split and process each character in the image
def split_digits_in_img(img_array, img_filename, x_list, y_list):
    step = img_cols // digits_in_img
    for i in range(digits_in_img):
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        y_list.append(char_to_int[img_filename[i]])
        #print(y_list[i])

# Assuming your training images are named with the label as the filename
img_filenames = os.listdir('training')

# Process each image file
for img_filename in img_filenames:
    if '.jpg' not in img_filename:
        continue
    #print(img_filename)
    img = load_img('./training/{0}'.format(img_filename), color_mode='grayscale')
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    split_digits_in_img(img_array, img_filename, x_list, y_list)

y_list = keras.utils.to_categorical(y_list, num_classes=36)
x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.1)

if os.path.isfile('cnn_model.h5'):
    model = models.load_model('cnn_model.h5')
    print('Model loaded from file.')
else:
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols // digits_in_img, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(rate=0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(36, activation='softmax'))
    print('New model created.')
 
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), batch_size=digits_in_img, epochs=epochs, verbose=1, validation_data=(np.array(x_test), np.array(y_test)))
 
loss, accuracy = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
 
model.save('cnn_model.h5')
