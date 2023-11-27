import requests
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import string
import numpy as np

# Create a reverse mapping from integers to characters
int_to_char = {idx: char for idx, char in enumerate(string.digits + string.ascii_uppercase)}

def split_digits_in_img(img_array, digits_in_img, img_cols):
    x_list = []
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
    return x_list

def predict_captcha(model, image_path, digits_in_img):
    img = load_img(image_path, color_mode='grayscale')
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    x_list = split_digits_in_img(img_array, digits_in_img, img_cols)

    verification_code = ''
    for i in range(digits_in_img):
        confidences = model.predict(np.array([x_list[i]]), verbose=0)
        result_class = np.argmax(confidences, axis=-1)
        predicted_char = int_to_char[result_class[0]]
        verification_code += predicted_char

    return verification_code

# use Session to keep cookie
session = requests.Session()

# load modile and predict the code
model = load_model('cnn_model.h5')
captcha_url = "此處使用動態生成CAPTCHA圖片的網址"
response = session.get(captcha_url, verify=False)
if response.status_code == 200:
    with open("code.jpg", "wb") as file:
        file.write(response.content)
    checkCode = predict_captcha(model, "code.jpg", 4)
    print('Predicted verification code:', checkCode)
else:
    print("Failed to retrieve CAPTCHA")

# login
login_url = "此處使用要登入的網頁" 
login_data = {
    'uid': "此處使用登入的帳號",
    'pwd': "此處使用登入的密碼",
    'checkCode': checkCode
}
login_response = session.post(login_url, data=login_data)
print(login_response.text)
url = "此處使用登入後的網頁"  # Replace with your URL
try:
    response = session.get(url)
    response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code.
    print(response.text) 
except requests.exceptions.HTTPError as errh:
    print ("Http Error:",errh)
except requests.exceptions.ConnectionError as errc:
    print ("Error Connecting:",errc)
except requests.exceptions.Timeout as errt:
    print ("Timeout Error:",errt)
except requests.exceptions.RequestException as err:
    print ("OOps: Something Else",err)