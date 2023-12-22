#시스템
import datetime
import os 
#딥러닝
from deep_sort_realtime.deepsort_tracker import DeepSort
from helper import create_video_writer, oriented_point, calculate_angle
from ultralytics import YOLO
#영상처리
import cv2

# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
LINE_THICKNESS = 2

# initialize the video capture object
video_cap = cv2.VideoCapture("./data/2.mp4")
# initialize the video writer object
writer = create_video_writer(video_cap, "./result/output.mp4")

# frame count
frame_cnt = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 프레임 속도 가지고 오기
frame_velo = int(video_cap.get(cv2.CAP_PROP_FPS))

# frame size
width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# oriented direction crosshead setup per frame
orient, orient_5 = oriented_point(width, height)

# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=50)

total_start = datetime.datetime.now()
while True:
    # 시작 시간을 fps 별로 계산
    start = datetime.datetime.now()
    
    ret, frame = video_cap.read() 
    
    # 만약 추가적인 프로세스가 없으면 while loop 그만 둠 
    if not ret:
        break 
    
    # yolo를 frame에 씌움 
    detections = model(frame)[0]
    
    #결과값 저장 
    results = [] 
    
    # detection loop
    for data in detections.boxes.data.tolist():
        # confidence 추출 : 정확도를 의미
        confidence = data[4]
        
        # confidence가 초기 세팅한 최소 confidence보다 높을떄만 추출
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue 
        
        # confidence가 최소 confidence보다 높을때 bounding box로 만들어줌
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        # results list에 프레임별로 저장 (x,y,w,h,confidence, class id)
        results.append([[xmin, ymin, xmax-xmin, ymax - ymin], confidence, class_id])
    # tracking 
    tracks = tracker.update_tracks(results, frame = frame)
    for track in tracks:
        if not track.is_confirmed():
            continue 
        # track id에 bounding 박스 라벨
        track_id = track.track_id
        ltrb = track.to_ltrb()
        
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]), 
        
        ## track id에 따라서 중심점 그리기
        box_x, box_y = (xmax+xmin) // 2, (ymax+ymin) // 2
        cv2.circle(frame, (box_x, box_y), radius = 5, color = GREEN, thickness = -1)
        cv2.putText(frame, f"x : {box_x}, y : {box_y}", (box_x+ 5, box_y -5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, GREEN, 1)
        
        ## 각도 구하기
        angle = calculate_angle((box_x, box_y), orient)
        cv2.line(frame, (box_x, box_y), orient, color = YELLOW, thickness= LINE_THICKNESS)
        
        ## track id에 따라서 boudning box 그리기
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, thickness = 2)
        cv2.rectangle(frame, (xmin, ymin -20), (xmin + 20, ymin), GREEN, thickness = -1)
        cv2.putText(frame, f"{track_id} : {angle:.1f}'", (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        
    # 마지막 fps를 계산해서 시간을 계산
    end = datetime.datetime.now()
    
    #프레임별 처리 시간 계산
    total = (end - start).total_seconds() 
    print(f"1프레임당 처리 시간 : {total*1000:.0f} msec")
    
    #fps 계산
    fps = f"FPS : {1/total:.2f}"
    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 8)
    
    #화살표 그려서 이미지 표시
    cv2.arrowedLine(frame, orient, orient_5 , color = RED, thickness = LINE_THICKNESS)
    
    #화면에 보여주기
    cv2.imshow("Frame", frame)
    writer.write(frame) #프레임별 객체저장 
    if cv2.waitKey(1) == ord("q"):
        break 
    
video_cap.release()
writer.release() 
cv2.destroyAllWindows()

#종료시간 
total_end = datetime.datetime.now() 

#전체 처리시간 계산 및 출력
print(f"프레임 수 : {frame_cnt}, 프레임 속도 : {frame_velo}")
print(f"기본 영상 시간 : {frame_cnt/frame_velo} sec | 전체 처리 시간 : {(total_end - total_start).total_seconds() } sec")