import cv2
from ultralytics import YOLO

# YOLO 모델 로드 (미리 학습된 모델 사용)
model = YOLO("C:/Users/ADMIN/Downloads/best (1).pt")
# 웹캠 초기화
cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    ret, frame = cap.read()

    if ret:
        
        results = model(frame)

       
        annotated_frame = results[0].plot()

        cv2.imshow("Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
       
        break

cap.release()
cv2.destroyAllWindows()