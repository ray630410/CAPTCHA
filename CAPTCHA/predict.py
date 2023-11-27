import numpy as np
import os
import string
import sys
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# Create a reverse mapping from integers to characters
int_to_char = {idx: char for idx, char in enumerate(string.digits + string.ascii_uppercase)}

img_rows = None
img_cols = None
digits_in_img = 6
model = None
np.set_printoptions(suppress=True, linewidth=150, precision=9, formatter={'float': '{: 0.9f}'.format})

def split_digits_in_img(img_array):
    x_list = list()
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
    return x_list


if os.path.isfile('cnn_model.h5'):
    model = models.load_model('cnn_model.h5')
else:
    print('No trained model found.')
    exit(-1)

img_filename = input('Varification code img filename: ')
img = load_img('./testImgs/{0}.jpg'.format(img_filename), color_mode='grayscale')
img_array = img_to_array(img)
img_rows, img_cols, _ = img_array.shape
x_list = split_digits_in_img(img_array)

varification_code = list()
for i in range(digits_in_img):
    confidences = model.predict(np.array([x_list[i]]), verbose=0)
    result_class = np.argmax(confidences, axis=-1)
    predicted_char = int_to_char[result_class[0]]
    varification_code.append(predicted_char)
    print('Digit {0}: Confidence=> {1}    Predict=> {2}'.format(i + 1, np.squeeze(confidences), predicted_char))

print('Predicted verification code:', ''.join(varification_code))
