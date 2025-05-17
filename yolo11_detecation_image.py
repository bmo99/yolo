import ultralytics
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('yolo11n.pt')

#이미지 경로 
image_path = "C:/Users/ADMIN/Downloads/test/test6.jpg"

#결과저장
results = model(image_path)

#이미지 보여주기
results[0].show()