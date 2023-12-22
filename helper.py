from numba import njit as njit 
import cv2
import math 


# 비디오 저장 
def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    return writer

# 비디오 저장 디렉토리 설정 정의
def createDirectory(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        print(f"SAVE PATH : {dir}")
    except OSError:
        print(f"ERROR : 디렉토리 생성하지 못하였음")
    
    return dir 


# # oriented_point 좌표 연산 및 처리 
@njit
def oriented_point(width, height):
    
    #frame별 shape 이용 
    center_x = width //2
    
    # 기준점 좌표에 화살표 만들기 : 아래에서 위로 향하는 걸로
    start_point = (center_x, height)
    end_point = (center_x, height - 50)
    
    return start_point, end_point


## 내적이용 각도 계산
@njit
def calculate_angle(dot1, dot2):
    # 두 점 간 상대적 좌표 계산
    dx = dot2[0] - dot1[0]
    dy = dot2[1] - dot1[1]
    
    # 아크탄젠트이용해서 각도 계산(라디안 단위)
    angle = math.atan2(dy, dx) 
    
    # 라디안을 degree로 변환
    angle = math.degrees(angle)
    
    return angle