# from realsense_camera import RealsenseCamera
from object_detection import ObjectDetection
import cv2
import time 
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import sys

# video_path = "D:/train_module/test_video/1.mp4"
video_path = "videotest.mp4"
object = ObjectDetection()

cap = cv2.VideoCapture(0)


def getContours(img, cThr=[100,100], showCanny=False, minArea=1000,filter=0, draw=False):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    
    kernel = np.ones((5,5))

    imgDial = cv2.dilate(imgCanny,kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)

    if showCanny: cv2.imshow("imgCanny",imgThre)

    contours, hiearchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    finalContours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea:
            peri = cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,0.02*peri,True)
            bbox = cv2.boundingRect(approx)
            
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append((len(approx),area,approx,bbox,contour))
            else:
                finalContours.append((len(approx),area,approx,bbox,contour))
    finalContours = sorted(finalContours, key=lambda x:x[1], reverse=True)
    if draw:
        for contour in finalContours:
            cv2.drawContours(img, [contour[4]], -1, (255,0,0), 10)
    
    return img, finalContours

def map_to_range(x, width):
    """
    Chuyển đổi giá trị x từ tọa độ pixel của khung hình (0 đến width)
    sang khoảng giá trị từ -100 đến 100.
    """
    if x < width // 2:  # Nếu x nằm trong khoảng từ 0 đến 320
        return -100 + (x / (width // 2)) * 100
    else:  # Nếu x nằm trong khoảng từ 320 đến 640
        return (x - width // 2) / (width // 2) * 100
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_filename = 'TaHaoNguyenTestVideo.mp4'
# codec = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_filename, codec, fps, (frame_width, frame_height))
def video():
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.resize(frame, (0,0), None, 0.5,0.5)
        # img, contours = getContours(frame,showCanny=False)
        # img_th = cv2.threshold(frame, 60, 255, cv2.THRESH_BINARY)[1]

        bboxes, class_ids, scores = object.detect(frame=frame, conf=0.5,device='0',classes=[0])

        end = time.time()
        if end - start == 0:
            continue
        # cv2.putText(frame, f"fps: {1/(end - start):.2f}", (100,100),cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 3)
        height, width = frame.shape[:2]
        # height, width = height//2, width//2
        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            #get center of the screen
            x, y, x2, y2 = bbox
            cx, cy = (x + x2) // 2, (y + y2) // 2
            color = object.colors[class_id]
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.rectangle(frame, (cx, y), (cx, y2), (0,255,0), 2)
            cv2.rectangle(frame, (x, cy), (x2, cy), (0,255,0), 2)
            # vi_tri = map_to_range(cx,width)
            # display name
            # class_name = object.classes[class_id]
            
            # cv2.putText(frame, f"{class_name}: {score}", (x, y - 10),
            # cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # cv2.putText(frame, f"{vi_tri:.2f}", (500,150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.imshow("frame",frame)
        # cv2.imshow("th",img_th)
        # out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def image():
    model = YOLO("CodeNapVoNaoMayThangCNTT.pt")
    # path = "TrainKhungRo_Ro/244.jpg"
    path = "HinhTest/503.jpg"
    results = model.predict(path,conf=0.4, device='0')
    result = results[0]
    result.show()
    img = cv2.imread(path)
    # y,x,z = img.shape
    # threshold = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)[1]
    # y,x,z = threshold.shape
    # print(x,y,z)
    # print(threshold)
    # gray = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.blur(gray,(3,3),5)
    # canny = cv2.Canny(blurred,100,250)

    cv2.imshow("img", img)
    # cv2.imshow("threshold", threshold)
    # cv2.imshow("blurred", blurred)
    # cv2.imshow("gray",gray)
    # cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("test video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def check_versions():
    python_version = sys.version
    print(f"Python Version: {python_version}")
    try:
        import pycuda.driver as cuda
        cuda.init()
        cuda_version = cuda.get_version()
        print(f"CUDA Version: {cuda_version}")
    except ImportError:
        print("pycuda is not installed or CUDA is not available.")

    try:
        import tensorflow as tf
        cudnn_version = tf.sysconfig.get_build_info().get("cudnn_version", "Unavailable")
        print(f"cuDNN Version: {cudnn_version}")
    except ImportError:
        print("TensorFlow is not installed or cuDNN information is unavailable.")

    try:
        import cv2
        opencv_version = cv2.__version__
        print(f"OpenCV Version: {opencv_version}")

        # Check if OpenCV was built with CUDA support
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"OpenCV is built with CUDA. Number of CUDA-enabled devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
            # Print CUDA device info
            cv2.cuda.printCudaDeviceInfo(0)
        else:
            print("OpenCV is not built with CUDA support.")
    except ImportError:
        print("OpenCV is not installed.")

if __name__ == "__main__":
    # video()
    # image()
    # test_video()
    check_versions()
