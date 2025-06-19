# Odd-Even Vehicle Interception Website v1 🚗🚙

Implementation of LNPR based on a website application using the YOLO model provided by Ultralytics. For now, this project is still in development stage 🛠️. So, there are still bugs due to the difference in the library version used when it was first created with the latest version. Some results can be seen below 📸.

## Features ✨

- 🇮🇩 Only Indonesian Vehicle
- 📝 Extract Text From Image
- 📅 Reporting Based On Date

## Screenshots 📷

![App Screenshot3](./documentation/older%20version/report.png)

## Run Locally 🖥️

Clone the project

``` bash
git clone https://github.com/brianajiks123/Odd-Even-Vehicle-Interception-Website-v1.git
```

Go to the project directory

``` bash
cd Odd-Even-Vehicle-Interception-Website-v1
```

Create .env file and store your OCR Space API 🔑

``` bash
OCR_API_KEY=K875xxx
```

Create Virtual Environment (make sure using Python version >= 3.8) 🐍

``` python
python -m venv venv
```

Activate Virtual Environment

``` python
venv\Scripts\activate
```

### OR

``` python
source venv\Scripts\activate
```

Install Library 📚

``` python
pip install -r requirements.txt
```

Start the server 🚀

``` python
python app.py
```

## Tech Stack 🛠️

**Client:** HTML, CSS, JavaScript 🌐

**Server:** Python, Flask, YOLO, OCR.space API 🖥️

## Acknowledgements 🙌

 - [Flask](https://flask.palletsprojects.com/en/stable/) 🧪
 - [You Only Look Once (YOLO)](https://docs.ultralytics.com/) 👁️
 - [OCR Space API](https://ocr.space/OCRAPI) 📖

## Authors ✍️

- [@brianajiks123](https://www.github.com/brianajiks123) 😎
