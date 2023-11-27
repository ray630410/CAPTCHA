import requests
import os
from random import randint
import time  # Import the time module

# Create a directory 'testImgs' if it does not exist
folder_path = './testImgs'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Function to download and save one CAPTCHA image
def download_captcha():
    random_code = randint(0, 10000)
    captcha_url = f"此處使用動態生成CAPTCHA圖片的網址"
    #captcha_url = f"https://eze8.npa.gov.tw/NpaE8ServerRWD/CheckCharImgServlet"
    response = requests.get(captcha_url, verify=False)
    if response.status_code == 200:
        file_path = os.path.join(folder_path, f"{random_code}.jpg")
        with open(file_path, "wb") as file:
            file.write(response.content)
        return True
    return False

# Download and save 100 images
success_count = 0
for _ in range(100):  # Adjust the range if you want more or fewer images
    if download_captcha():
        success_count += 1
    time.sleep(3)  # Wait for 3 seconds before downloading the next image

print(f"Successfully downloaded {success_count} images.")
