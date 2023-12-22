### 기능 구현  
- Yolov8를 이용해서 객체 탐지
- DeepSort 알고리즘 이용해서 객체 라벨링 구성
- 중심 angle 측정
- Visualization 
- + 추가 중

### 처리영상
![output (1)](https://github.com/dablro12/robotics/assets/54443308/10435d23-3b66-48c4-b91c-cdd68b6b367f)



### ROBTOICS file list 
- detect.py : 실행 파일
- detect_real_time.py : 처리 최적화 검사 파일
- helper : 모델 필요 수학 함수 및 셋업 함수 저장 파일 

### Robotics Folder list 
- data : 테스트 데이터 폴더
- model : 모델 폴더
- result : 결과 저장 폴더
- setup : ARM based gpu 가속 기능 확인 폴더 for mac 


## Reference Code Link
- Yolov8 with DeepSORT Algorithms : https://thepythoncode.com/article/real-time-object-tracking-with-yolov8-opencv
