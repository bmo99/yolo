import cv2
from ultralytics import YOLO

# 커스텀 모델 로드
model = YOLO('C:/Users/ADMIN/bmo/runs/detect/train/weights/best.pt')    

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