import cv2
from detect_code.main import img_detect
import time
from collections import Counter

path = 'rtsp://admin:hdu417417@192.168.4.13/Streaming/Channels/101'
path2 = './digitalsets/demo.avi'
path3 = 'rtsp://admin:hdu417417@192.168.1.9/Streaming/Channels/101'
cap = cv2.VideoCapture(path)
print(cap.isOpened())
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:",fps)  # 25帧/s

interval = 40
digit_queue = []
count = 0
count2 = 0
begin = time.time()

# 对于读取存储的视频文件,
while cap.isOpened():
    ret, frame = cap.read()

    result = img_detect(frame)

    # 判断队列中数组组成

    if len(digit_queue) == interval:
        if count2 == interval:
            ret_count = Counter(digit_queue).most_common(2)
            count2 = 0
        first_digit = ret_count[0][0]
        # 把首帧删除
        digit_queue.pop(0)
        if count == interval and first_digit != 0:
            count = 0
            print(first_digit)
            digit_queue = []
    digit_queue.append(result)
    count += 1
    count2 += 1
    # cv2.putText(frame, str(result), (1150, 750), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
    img = cv2.resize(frame, (1200, 500), interpolation=cv2.INTER_LINEAR)
    k = cv2.waitKey(10)
    cv2.imshow('img', img)
    # q键退出
    if k & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
