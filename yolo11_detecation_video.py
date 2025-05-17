import ultralytics
from ultralytics import YOLO
import cv2


# YOLO 모델 로드
model = YOLO('yolo11n.pt')

# 동영상 경로
video_path = "C:/Users/ADMIN/Downloads/test/test_video.mp4"
cap = cv2.VideoCapture(video_path)  # OPENCV 통해서 동영상 파일 열기

# 비디오 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 동영상 코덱 설정
out = cv2.VideoWriter("output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 동영상이 끝나면 종료

    # YOLO 객체 탐지 수행
    results = model(frame)

    # 탐지 결과를 프레임에 적용
    annotated_frame = results[0].plot()  # 바운딩 박스와 레이블이 그려진 프레임 생성

    # 결과 영상 저장
    out.write(annotated_frame)


