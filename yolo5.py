# import torch
# import cv2
# import numpy as np
# import time

# def POINTS(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         colorsBGR = [x, y]
#         print(colorsBGR)

# cv2.namedWindow('ROI')
# cv2.setMouseCallback('ROI', POINTS)

# cap=cv2.VideoCapture('vid1.mp4')

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# num_rectangles = 0
# isUsed = False
# # 수정: 추가된 기능을 반영하기 위해 이전에 사용한 변수 초기화
# start_time = time.time()
# no_rectangle_start_time = None

# while True:
#     ret,frame=cap.read()
#     if not ret:
#         break
    
#     frame=cv2.resize(frame,(1020,500))
#     results = model(frame)
#     num_rectangles = 0
    
#     for index, row in results.pandas().xyxy[0].iterrows():
#         xmin = int(row['xmin'])
#         ymin = int(row['ymin'])
#         xmax = int(row['xmax'])
#         ymax = int(row['ymax'])
#         class_name = row['name']
        
#         if class_name == 'person' and xmin > 297 and xmax < 690 and ymin > 64 and ymax < 273:
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             num_rectangles += 1
            
#             # 추가: 인식된 사람이 영역 내부에 있으면, isUsed를 True로 바꿔준다
#             if time.time() - start_time > 5:
#                 isUsed = True
            
#             # 추가: 사각형이 인식된 시간 초기화
#             no_rectangle_start_time = None
            
#     # 추가: 인식된 사각형이 없을 때
#     if num_rectangles == 0:
#         # 추가: no_rectangle_start_time이 None이면 초기화
#         if no_rectangle_start_time is None:
#             no_rectangle_start_time = time.time()
        
#         # 추가: 인식된 사각형이 없는 상태가 3초 이상 유지되면 isUsed를 다시 False로 바꿔준다
#         if time.time() - no_rectangle_start_time > 3 and isUsed:
#             isUsed = False
            
#             # 추가: 다시 False로 변경된 시간 초기화
#             start_time = time.time()
#     if isUsed:
#         cv2.putText(frame, 'Person is inside', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#     else:
#         cv2.putText(frame, 'Person is not inside', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
#     # Print the number of rectangles on the screen
#     cv2.putText(frame, f'Number of rectangles: {num_rectangles}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     cv2.imshow("ROI",frame)
#     if cv2.waitKey(1)&0xFF==27:
#         break
# cap.release()
# cv2.destroyAllWindows()


# 이 위로 중앙부의 사람만 사각형으로 표시/갯수 알려줌/5초이상 영역 내부에 사람이 인식되면 isUsed를 True로 바꿔줌/3초이상 영역 내부에 사람이 인식되지 않으면 isUsed를 False로 바꿔줌

# import torch
# import cv2
# import numpy as np
# import time

# def POINTS(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         colorsBGR = [x, y]
#         print(colorsBGR)

# cv2.namedWindow('ROI')
# cv2.setMouseCallback('ROI', POINTS)

# cap=cv2.VideoCapture('vid1.mp4')
# #cap=cv2.VideoCapture(0)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# num_rectangles = 0
# isUsed = False
# start_time = time.time()
# no_rectangle_start_time = None

# while True:
#     ret,frame=cap.read()
#     if not ret:
#         break
    
#     frame=cv2.resize(frame,(1020,500))
#     num_rectangles = 0  # 인식된 사람 수 초기화
#     results = model(frame)
    
#     for index, row in results.pandas().xyxy[0].iterrows():
#         x1 = int(row['xmin'])
#         y1 = int(row['ymin'])
#         x2 = int(row['xmax'])
#         y2 = int(row['ymax'])
#         name = row['name']
#         if name == 'person':  # 사람만 인식하도록 수정
#             if x1 >= 273 and y1 >= 64 and x2 <= 690 and y2 <= 297:  # 좌표값이 화면의 [64:297, 273:690] 범위 안에 속하는 경우에만 사각형을 그리도록 수정
#                 cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
#                 num_rectangles += 1  # 인식된 사람 수 증가

#                 #추가: 인식된 사람이 영역 내부에 있으면, isUsed를 True로 바꿔준다
#                 if time.time() - start_time > 5:
#                     isUsed = True
                
#                 # 추가: 사각형이 인식된 시간 초기화
#                 no_rectangle_start_time = None

#     # 추가: 인식된 사각형이 없을 때
#     if num_rectangles == 0:
#         # 추가: no_rectangle_start_time이 None이면 초기화
#         if no_rectangle_start_time is None:
#             no_rectangle_start_time = time.time()
        
#         # 추가: 인식된 사각형이 없는 상태가 3초 이상 유지되면 isUsed를 다시 False로 바꿔준다
#         if time.time() - no_rectangle_start_time > 3 and isUsed:
#             isUsed = False
            
#             # 추가: 다시 False로 변경된 시간 초기화
#             start_time = time.time()

#     if isUsed:
#         cv2.putText(frame, 'isUsed : true', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#     else:
#         cv2.putText(frame, 'isUsed : false', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    
#     # [64:297, 273:690] 범위를 붉은색 얇은 선으로 그림
#     cv2.rectangle(frame, (273, 64), (690, 297), (0, 0, 255), 1)

#     # Print the number of rectangles on the screen
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(frame, f'Number of peoples: {num_rectangles}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     cv2.imshow("ROI",frame)
#     if cv2.waitKey(1)&0xFF==27:
#         break
# cap.release()
# cv2.destroyAllWindows()


#이 위로 중앙구역만 인식/isUsed가 True면 빨간색 False면 초록색으로 표시


import torch
import cv2
import numpy as np
import time

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)

#cap=cv2.VideoCapture('vid1.mp4')
cap=cv2.VideoCapture('vid4.mp4')
#cap=cv2.VideoCapture(0)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
num_rectangles = 0
isUsed = False
start_time = time.time()
no_rectangle_start_time = None

while True:
    ret,frame=cap.read()
    if not ret:
        break
    
    frame=cv2.resize(frame,(1920,1080))
    num_rectangles = 0  # 인식된 사람 수 초기화
    results = model(frame)
    
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        name = row['name']
        if name == 'person':  # 사람만 인식하도록 수정
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            num_rectangles += 1  # 인식된 사람 수 증가

            # 추가: 인식된 사각형이 영역 내부에 있으면, isUsed를 True로 바꿔준다
            if 273 <= (x1+x2)/2 <= 690 and 64 <= (y1+y2)/2 <= 297:
                if time.time() - start_time > 5:
                    isUsed = True
                
                # 추가: 사각형이 인식된 시간 초기화
                no_rectangle_start_time = None

    # 추가: 인식된 사각형이 없을 때
    if num_rectangles == 0:
        # 추가: no_rectangle_start_time이 None이면 초기화
        if no_rectangle_start_time is None:
            no_rectangle_start_time = time.time()
        
        # 추가: 인식된 사각형이 없는 상태가 3초 이상 유지되면 isUsed를 다시 False로 바꿔준다
        if time.time() - no_rectangle_start_time > 3 and isUsed:
            if not any([273 <= (x1+x2)/2 <= 690 and 64 <= (y1+y2)/2 <= 297 for index, row in results.pandas().xyxy[0].iterrows() if row['name'] == 'person']):
                isUsed = False
                
                # 추가: 다시 False로 변경된 시간 초기화
                start_time = time.time()

    if isUsed:
        cv2.putText(frame, 'isUsed : true', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'isUsed : false', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    
    cv2.rectangle(frame, (273, 64), (690, 297), (0, 0, 255), 1)

    # Print the number of rectangles on the screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Number of peoples: {num_rectangles}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("ROI",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

