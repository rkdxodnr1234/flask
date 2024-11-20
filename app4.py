from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np
import os
import tempfile
import uuid
import joblib
import tensorflow as tf

app = Flask(__name__)

# 모델 로드
MODEL_DIR = "C:/Users/ahsld/Desktop/flask/models"

# LSTM 모델 및 스케일러 로드
lstm_model = tf.keras.models.load_model('lstm_model.h5')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# 현재 스크립트 위치를 기준으로 업로드 폴더 설정
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/assets/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 클래스 이름 지정. 학습 클래스 이름이 너무 길어서
CLASS_NAMES = {0: 'breaker', 1: 'pink', 2: 'red'}

# 클래스별 색상 지정
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 255, 0), 2: (255, 0, 0)}

def load_yolo_model(model_filename):
    """선택한 YOLO 모델 로드"""
    model_path = os.path.join(MODEL_DIR, model_filename)
    return YOLO(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/charts')
def charts():
    return render_template('charts.html')

@app.route('/tables')
def tables():
    return render_template('tables.html')

@app.route('/yolo_index')
def yolo_index():
    return render_template('YOL_index.html')

@app.route('/lstm_index')
def lstm_index():
    return render_template('lstm_index.html')

# YOLO 예측 라우트
@app.route('/predict_yolo', methods=['POST'])
def predict_yolo():
    selected_model = request.form.get('model')
    model = load_yolo_model(selected_model)
    
    file = request.files.get('file')
    if not file:
        return "파일이 없습니다.", 400
 
    model_path = os.path.join("C:/Users/ahsld/Desktop/flask/models", selected_model)
    
    # UUID 기반 파일명으로 저장
    ext = file.filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    if file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
        img = Image.open(file_path)
        results = model(img)
        
        # 탐지한 객체 바운딩박스 그리기
        annotated_img = np.array(img)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]  # 클래스 이름 얻기

                color = CLASS_COLORS.get(class_id, (255, 255, 255))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_img, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 결과 이미지 저장
        result_image_filename = 'result_' + unique_filename
        result_path = os.path.join(UPLOAD_FOLDER, result_image_filename)
        cv2.imwrite(result_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

        return render_template('YOLO_result.html', result_image=result_image_filename, model_name=selected_model)


    # 동영상 파일 처리
    elif file.filename.lower().endswith(('mp4', 'avi', 'mov')):
        cap = cv2.VideoCapture(file_path)
        result_video_filename = f'result_{unique_filename}'
        output_path = os.path.join(UPLOAD_FOLDER, result_video_filename)
        
        # MP4 코덱 설정 (H264 코덱을 사용하여 mp4로 저장)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 탐지 수행 및 결과 그리기
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = model(img)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = CLASS_NAMES.get(class_id, "Unknown")# 바꾼 코드

                    color = (255, 255, 255) if class_id == 0 else (0, 165, 255) if class_id == 1 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)

        cap.release()
        out.release()

        return render_template('YOLO_result.html', result_video=result_video_filename, model_name=selected_model)

    else:
        return "유효한 이미지 파일이 아닙니다.", 400

@app.route('/yolo_result')
def yolo_result():
    result = request.args.get('result', None)
    return render_template('YOLO_result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
