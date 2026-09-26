# 📸 Face Recognition Attendance System

An AI-based Face Recognition Attendance System built using Python, OpenCV, CNN, TensorFlow/Keras, and Streamlit.

## 📌 About the Project

This project uses face recognition to identify registered users and automatically mark their attendance.

The application provides a simple Streamlit web interface where a user can capture their image using the camera. The system detects the face, predicts the person's identity using a trained CNN model, and records the attendance with the date and time.

## ✨ Features

- 📷 Camera-based face capture
- 👤 Face detection using OpenCV
- 🤖 Face recognition using CNN
- 📊 Prediction confidence display
- 📝 Automatic attendance marking
- 📅 Date and time recording
- 🌐 Streamlit web interface
- 🚫 Prevents duplicate attendance on the same day

## 🛠️ Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Pandas
- Streamlit
- Jupyter Notebook
- Haar Cascade Classifier

## ⚙️ How It Works

1. The user captures an image using the camera.
2. OpenCV detects the face.
3. The detected face is preprocessed.
4. The trained CNN model predicts the person's identity.
5. The prediction confidence is displayed.
6. If the confidence meets the required threshold, attendance is recorded.
7. Attendance is stored with the person's name, date, time, and status.

## 📂 Project Structure

```text
face-recognition-attendance/
│
├── app.py
├── attendance.csv
├── final_model.h5
├── haarcascade_frontalface_default (1).xml
│
├── attendance.ipynb
├── consolidate_data.ipynb
├── data_collect.ipynb
├── Face_Detection (1).ipynb
└── recognize.ipynb
