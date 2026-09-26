import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from keras.models import load_model


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="📸",
    layout="centered"
)


# ==========================================
# LOAD FACE DETECTOR AND MODEL
# ==========================================

classifier = cv2.CascadeClassifier(
    "haarcascade_frontalface_default (1).xml"
)

model = load_model("final_model.h5")


# IMPORTANT:
# This order must match the LabelEncoder
# used during your training.

labels = [
    "Sapna",
    "Simran",
    "dibiya"
]


# ==========================================
# PREPROCESS FACE
# ==========================================

def preprocess(img):

    # Convert to grayscale
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Resize exactly like training
    img = cv2.resize(
        img,
        (100, 100)
    )

    # Histogram equalization
    img = cv2.equalizeHist(img)

    # Add channel dimension
    img = img.reshape(
        1,
        100,
        100,
        1
    )

    # Normalize
    img = img / 255.0

    return img


# ==========================================
# MARK ATTENDANCE
# ==========================================

def mark_attendance(name):

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Load existing attendance
    if os.path.exists("attendance.csv"):

        df = pd.read_csv(
            "attendance.csv"
        )

    else:

        df = pd.DataFrame(
            columns=[
                "Name",
                "Date",
                "Time",
                "Status"
            ]
        )

    # Check duplicate attendance
    already_present = (
        (df["Name"] == name) &
        (df["Date"] == date)
    ).any()

    if already_present:
        return False

    # New attendance record
    new_record = pd.DataFrame([{
        "Name": name,
        "Date": date,
        "Time": time,
        "Status": "Present"
    }])

    # Add record
    df = pd.concat(
        [df, new_record],
        ignore_index=True
    )

    # Save
    df.to_csv(
        "attendance.csv",
        index=False
    )

    return True


# ==========================================
# WEB APP
# ==========================================

st.title(
    "📸 Face Recognition Attendance"
)

st.write(
    "Capture your face using the camera "
    "to mark attendance."
)


# ==========================================
# CAMERA
# ==========================================

image = st.camera_input(
    "Take a picture"
)


# ==========================================
# PROCESS CAMERA IMAGE
# ==========================================

if image is not None:

    # Read image bytes
    bytes_data = image.getvalue()

    np_array = np.frombuffer(
        bytes_data,
        np.uint8
    )

    frame = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )


    # ======================================
    # FACE DETECTION
    # Same settings as original code
    # ======================================

    faces = classifier.detectMultiScale(
        frame,
        scaleFactor=1.3,
        minNeighbors=5
    )


    # ======================================
    # NO FACE
    # ======================================

    if len(faces) == 0:

        st.error(
            "❌ No face detected. "
            "Please take another picture."
        )


    else:

        st.success(
            f"👤 {len(faces)} face detected."
        )


        # ==================================
        # SELECT LARGEST FACE
        # ==================================

        x, y, w, h = max(
            faces,
            key=lambda f: f[2] * f[3]
        )


        # Crop detected face
        face = frame[
            y:y+h,
            x:x+w
        ]


        # ==================================
        # SHOW FACE USED BY MODEL
        # ==================================

        st.subheader(
            "🔍 Face used for recognition"
        )

        st.image(
            cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )


        # ==================================
        # PREPROCESS
        # ==================================

        processed_face = preprocess(
            face
        )


        # ==================================
        # MODEL PREDICTION
        # ==================================

        prediction = model.predict(
            processed_face,
            verbose=0
        )[0]


        # ==================================
        # SHOW ALL PROBABILITIES
        # ==================================

        st.subheader(
            "📊 Model Prediction"
        )

        for i in range(len(labels)):

            st.write(
                f"**{labels[i]}:** "
                f"{prediction[i] * 100:.2f}%"
            )


        # ==================================
        # GET PREDICTED PERSON
        # ==================================

        predicted_class = np.argmax(
            prediction
        )

        name = labels[
            predicted_class
        ]

        confidence = (
            float(
                prediction[predicted_class]
            ) * 100
        )


        st.write(
            f"### 👤 Predicted: {name}"
        )

        st.write(
            f"### 🎯 Confidence: "
            f"{confidence:.2f}%"
        )


        # ==================================
        # ATTENDANCE
        # ==================================

        if confidence >= 70:

            result = mark_attendance(
                name
            )


            if result:

                st.success(
                    f"🎉 Attendance marked for "
                    f"**{name}**"
                )

            else:

                st.info(
                    f"ℹ️ **{name}** is already "
                    "marked present today."
                )


        else:

            st.warning(
                "⚠️ Confidence is too low. "
                "Attendance was not marked."
            )


# ==========================================
# TODAY'S ATTENDANCE
# ==========================================

st.divider()

st.subheader(
    "📋 Today's Attendance"
)


if os.path.exists("attendance.csv"):

    df = pd.read_csv(
        "attendance.csv"
    )


    today = datetime.now().strftime(
        "%Y-%m-%d"
    )


    today_attendance = df[
        df["Date"] == today
    ]


    if len(today_attendance) > 0:

        st.dataframe(
            today_attendance,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No attendance marked today."
        )

else:

    st.info(
        "No attendance records yet."
    )