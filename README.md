# CAPTCHA##  網頁驗證碼(CAPTCHA)辨識模型訓練 (CNN)
### 摘要
這個報告旨在開發一個基於卷積神經網絡（CNN）的模型，用於自動識別網頁驗證碼（CAPTCHA），進而實現自動登入特定網站並抓取數據。首先，我通過爬蟲技術從目標網站下載CAPTCHA圖片，以建立訓練數據集。接著，我修改了網路上別人設計和訓練的一個深度學習模型，該模型能夠從圖片中識別出驗證碼。我利用它對我的目標站上爬下來的圖片進行了訓練及圖片預測測試。然後將預測的程式整合到自動登入的腳本中。最終，我成功實現了自動識別驗證碼並登入網站的功能，這對於我想要自動化數據收集和處理具有很大的意義。本報告提供了詳細的方法、實踐用的程式碼，以及為了完成這個目標還有可能遇到的其他許多挑戰。
### 動機
許多網站為了避免非人為操作的存取頻繁增加了網站的流量負擔，因此建立了驗證碼機制來限制網站只有人可以進行操作。但是身為一個想善用資訊網路替我們服務的現代人，我們常常對於一個自己擁有帳號密碼的網站，有自動前往讀取資料的需求。
例如：我們有數個銀行的存款帳戶跟信用卡，同時也有網路銀行的帳密，我們希望能自動將各銀行帳戶中的存款，信用卡消費總額同步到我們的個人資產負債表中方便一目瞭然的管理個人資產。
### 步驟
- 爬取並下載目標網站中的CAPTCHA圖片方便進行訓練
- 人工標記圖片
- 撰寫訓練程式並將訓練好的模型儲存下來
- 撰寫預測程式並載入訓練好的模型來對圖片進行辨識
- 整合成自動登入網站的程式
### 簡單認識網頁驗證碼(CAPTCHA)
#### 目標網頁範例：
https://eze8.npa.gov.tw/NpaE8ServerRWD/CL_Query.jsp
#### 用動態生成CAPTCHA圖片的網址範例：
https://eze8.npa.gov.tw/NpaE8ServerRWD/CheckCharImgServlet
### 步驟實作
#### 爬取並下載目標網站中的CAPTCHA圖片方便進行訓練
- 載入必要的函式庫
```python=
import requests
import os
from random import randint
import time
```
- 建立下載圖片儲存的資料夾
```python=
# Create a directory 'testImgs' if it does not exist
folder_path = './testImgs'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
```
- 建立下載圖片並儲存的函式
```python=
# Function to download and save one CAPTCHA image
def download_captcha():
    random_code = randint(0, 10000)
    captcha_url = f"此處使用動態生成CAPTCHA圖片的網址"
    response = requests.get(captcha_url, verify=False)
    if response.status_code == 200:
        file_path = os.path.join(folder_path, f"{random_code}.jpg")
        with open(file_path, "wb") as file:
            file.write(response.content)
        return True
    return False
```
- 呼叫下載圖片並儲存的函式100次，每次間隔3秒避免被當作攻擊
```python=
# Download and save 100 images
success_count = 0
for _ in range(100):  # Adjust the range if you want more or fewer images
    if download_captcha():
        success_count += 1
    time.sleep(3)  # Wait for 3 seconds before downloading the next image
print(f"Successfully downloaded {success_count} images.")
```
#### 人工標記圖片

#### 撰寫訓練程式並將訓練好的模型儲存下來
- 載入必要的函式庫
```python=
import string
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
```
- 定義一些訓練時要使用的參數
```python=
epochs = 10       # Number of epochs for training
img_rows, img_cols = None, None  # Will be determined based on the loaded image
digits_in_img = 4 # CAPTCHA length is now 4 characters
num_classes = 36  # 26 English alphabets + 10 digits
x_list, y_list = [], []
x_train, y_train, x_test, y_test = [], [], [], []
```
- 建立數字字元及大寫英文字元對應到數字的對應表
```python=
# Updated mapping for alphanumeric characters
char_to_int = {char: idx for idx, char in enumerate(string.digits + string.ascii_uppercase)}
```
- 定義分割圖片中的字元，並將圖像放進x_lis，檔名中的實際字元轉換的數字放進y_list
```python=
# Split and process each character in the image
def split_digits_in_img(img_array, img_filename, x_list, y_list):
    step = img_cols // digits_in_img
    for i in range(digits_in_img):
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        y_list.append(char_to_int[img_filename[i]])
        #print(y_list[i])
```
- 將training資料夾中的所有圖片分割影像資料放進x_list，從檔名取得字元對應的數字放進y_list
```python=
# Assuming your training images are named with the label as the filename
img_filenames = os.listdir('training')
# Process each image file
for img_filename in img_filenames:
    if '.jpg' not in img_filename:
        continue
    print(img_filename)
    img = load_img('./training/{0}'.format(img_filename), color_mode='grayscale')
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    split_digits_in_img(img_array, img_filename, x_list, y_list)
```
- 將y_list轉換成二維陣列，本來每一個元素就變成一個由0,1組成的，36個元素的陣列
```python=
y_list = keras.utils.to_categorical(y_list, num_classes=36)
x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.1)
```
- 載入或是建立模型
```python=
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
```
---
- 訓練及測試模型，並輸出準確度
```python=
model.fit(np.array(x_train), np.array(y_train), batch_size=digits_in_img, epochs=epochs, verbose=1, validation_data=(np.array(x_test), np.array(y_test)))
# epochs is trainning time, here is 10
loss, accuracy = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
```
- 將模型存檔
```python=
model.save('cnn_model.h5')
```
#### 撰寫預測程式並載入訓練好的模型來對圖片進行辨識
- 載入必要的函式庫
```python=
import numpy as np
import os
import string
import sys
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
```
- 建立數字對應到數字字元及大寫英文字元的對應表
```python=
# Create a reverse mapping from integers to characters
int_to_char = {idx: char for idx, char in enumerate(string.digits + string.ascii_uppercase)}
```
- 定義一些預測時要使用的參數
```python=
img_rows = None
img_cols = None
digits_in_img = 4
model = None
np.set_printoptions(suppress=True, linewidth=150, precision=9, formatter={'float': '{: 0.9f}'.format})
```
- 定義分割圖片中的字元，並將圖像放進x_lis並回傳
```python=
def split_digits_in_img(img_array):
    x_list = list()
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
    return x_list
```
- 載入模型，不存在模型檔則報錯
```python=
if os.path.isfile('cnn_model.h5'):
    model = models.load_model('cnn_model.h5')
else:
    print('No trained model found.')
    exit(-1)
```
- 讓使用者輸入要測試的圖片主檔名
```python=
img_filename = input('Varification code img filename: ')
```
- 載入圖檔並將分割的四個字元圖片放入x_list
```python=
img = load_img('./testImgs/{0}.jpg'.format(img_filename), color_mode='grayscale')
img_array = img_to_array(img)
img_rows, img_cols, _ = img_array.shape
x_list = split_digits_in_img(img_array)
```
- 預測x_list中的每個字元圖片的數字並轉換成對應的字元
```python=
varification_code = list()
for i in range(digits_in_img):
    confidences = model.predict(np.array([x_list[i]]), verbose=0)
    result_class = np.argmax(confidences, axis=-1)
    predicted_char = int_to_char[result_class[0]]
    varification_code.append(predicted_char)
    print('Digit {0}: Confidence=> {1}    Predict=> {2}'.format(i + 1, np.squeeze(confidences), predicted_char))
```
- 輸出預測的字元(CAPTCHA驗證碼)
```python=
print('Predicted verification code:', ''.join(varification_code))
```
#### 整合成自動登入網站的程式
- 載入必要的函式庫
```python==
import requests
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import string
import numpy as np
```
- 建立數字對應到數字字元及大寫英文字元的對應表
```python=
# Create a reverse mapping from integers to characters
int_to_char = {idx: char for idx, char in enumerate(string.digits + string.ascii_uppercase)}
```
- 定義分割圖片中的字元，並將圖像放進x_lis並回傳
```python=
def split_digits_in_img(img_array, digits_in_img, img_cols):
    x_list = []
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
    return x_list
```
- 建立預測CAPTCHA驗證碼並回傳的函式
```python=
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
```
- 建立要求網頁的session以確保驗證碼與登入頁同一session
```python=
# use Session to keep cookie
session = requests.Session()
```
- 載入模型
```python=
# load model and predict the code
model = load_model('cnn_model.h5')
```
- 爬取驗證碼並存檔
```python=
captcha_url = "此處使用動態生成CAPTCHA圖片的網址"
response = session.get(captcha_url, verify=False)
if response.status_code == 200:
    with open("code.jpg", "wb") as file:
        file.write(response.content)
```
- 呼叫預測CAPTCHA驗證碼並回傳的函式以取得預測的CAPTCHA驗證碼
```python=
    checkCode = predict_captcha(model, "code.jpg", 4)
    print('Predicted verification code:', checkCode)
else:
    print("Failed to retrieve CAPTCHA")
```
- 使用預測的CAPTCHA驗證碼登入網頁
```python=
# login
login_url = "此處使用要登入的網頁" 
login_data = {
    'uid': "此處使用登入的帳號",
    'pwd': "此處使用登入的密碼",
    'checkCode': checkCode
}
login_response = session.post(login_url, data=login_data)
```
- 輸出執行登入的結果
```python=
print(login_response.text)
```
- 取得登入後有權限造訪的網頁的結果
```python=
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
```

**[參考資料]**
- 用TensorFlow+Keras訓練辨識驗證碼的CNN模型
https://notes.andywu.tw/2019/%E7%94%A8tensorflowkeras%E8%A8%93%E7%B7%B4%E8%BE%A8%E8%AD%98%E9%A9%97%E8%AD%89%E7%A2%BC%E7%9A%84cnn%E6%A8%A1%E5%9E%8B/
