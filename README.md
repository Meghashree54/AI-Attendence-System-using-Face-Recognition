# Face-recognition-Attendance-System-Project

This project is rebuilt from scratch with the same core workflow as the reference Flask project:

- Add user by capturing face images from webcam
- Train a KNN model from saved face samples
- Start attendance recognition from webcam
- Store attendance in daily CSV files
- List and delete registered users

# Project Structure

- ```app.py```: Main Flask application
- ```templates/home.html```: Dashboard (attendance + add user)
- ```templates/listusers.html```: User list and delete page
- ```static/faces/```: Dataset folder in Name_ID/ structure
- ```static/face_recognition_model.pkl```: Trained model file (auto-generated)
- ```Attendance/Attendance-MM_DD_YY.csv```: Daily attendance log (auto-generated)


