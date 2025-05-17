import cv2
from ultralytics import YOLO
import yaml
from glob import glob
import zipfile
import os

def main():
    data = {
        'train': 'C:/Users/ADMIN/Downloads/RockPaperScissors_Data/train/images/',
        'val': 'C:/Users/ADMIN/Downloads/RockPaperScissors_Data/valid/images/',
        'test': 'C:/Users/ADMIN/Downloads/RockPaperScissors_Data/test/images',
        'names': ['Paper', 'Rock', 'Scissors'],
        'nc': 3
    }

    # YAML 파일 작성
    with open('C:/Users/ADMIN/Downloads/RockPaperScissors_Data/RockPaperScissors_Data.yaml', 'w') as f:
        yaml.dump(data, f)

    with open('C:/Users/ADMIN/Downloads/RockPaperScissors_Data/RockPaperScissors_Data.yaml', 'r') as f:
        print(yaml.safe_load(f))

    # YOLO 모델 로드
    model = YOLO('yolov5s.pt')

    # 모델 학습
    model.train(data='C:/Users/ADMIN/Downloads/RockPaperScissors_Data/RockPaperScissors_Data.yaml', epochs=30,patience=5, imgsz=416)

    # 테스트 이미지 리스트 가져오기
    test_image_list = glob('C:/Users/ADMIN/Downloads/RockPaperScissors_Data/test/images/*')
    print(len(test_image_list))
    test_image_list.sort()

    # 테스트 이미지 출력
    for i in range(len(test_image_list)):
        print('i = ', i, test_image_list[i])

    # 모델 예측 결과 저장
    results = model(source='C:/Users/ADMIN/Downloads/RockPaperScissors_Data/test/images/', save=True)
 

if __name__ == '__main__':
    main()

