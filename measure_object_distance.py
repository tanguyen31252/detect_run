#http://www.pysource.com

from realsense_camera import RealsenseCamera
from object_detection import ObjectDetection
import cv2
import numpy as np

# Create the Camera object
# camera = RealsenseCamera()

# Create the Object Detection object
object_detection = ObjectDetection()

def map_to_range(x, width):
    """
    Chuyển đổi giá trị x từ tọa độ pixel của khung hình (0 đến width)
    sang khoảng giá trị từ -100 đến 100.
    """
    if x < width // 2:  # Nếu x nằm trong khoảng từ 0 đến 320
        return -100 + (x / (width // 2)) * 100
    else:  # Nếu x nằm trong khoảng từ 320 đến 640
        return (x - width // 2) / (width // 2) * 100

def getContours(img, cThr=[100,100], showCanny=False, minArea=1000,filter=0, draw=False):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    
    kernel = np.ones((5,5))

    imgDial = cv2.dilate(imgCanny,kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)
    imgThreshold = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]

    if showCanny: 
        # cv2.imshow("imgGray",imgGray)
        # cv2.imshow("imgBlur",imgBlur)
        # cv2.imshow("imgCanny",imgCanny)
        # cv2.imshow("imgDial",imgDial)
        cv2.imshow("imgThre",imgThre)
        cv2.imshow("imgThreshold",imgThreshold)

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
    
    # return imgGray, imgBlur, imgCanny, imgDial, imgThre

def dao_anh(img):
    return 255-img
def log_transform(img, c=1):
    """
    Hàm chuyển đổi logarit (Logarithmic transformation).
    
    Parameters:
    img: Ảnh đầu vào (grayscale hoặc RGB).
    c: Hằng số tỷ lệ (mặc định là 1).
    
    Returns:
    Ảnh sau khi chuyển đổi logarit.
    """
    # Chuyển đổi sang kiểu float32 để tránh hiện tượng bị tràn giá trị
    img_float = np.float32(img)
    
    # Áp dụng công thức chuyển đổi logarit
    log_img = c * np.log(1 + img_float)
    
    # Chuẩn hóa ảnh về lại kiểu 8-bit
    log_img = np.uint8(255 * log_img / np.max(log_img))
    
    return log_img

def detect_ro():
    while True:
        # Get color_image from realsense camera
        ret, color_image, depth_image = camera.get_frame_stream()
        height, width, _ = color_image.shape
        # Get the object detection
        # getContours(color_image, draw=True, minArea=50000,filter=4)
        bboxes, class_ids, score = object_detection.detect(color_image,640, device='0',classes=[0,2])
        cv2.rectangle(color_image, (0,height-150), (250,height), (255,255,255), -1)
        cv2.rectangle(color_image, (width,height-150), (width-250,height), (255,255,255), -1)
        for bbox, class_id, score in zip(bboxes, class_ids, score):
            x, y, x2, y2 = bbox
            color = object_detection.colors[class_id]
            cv2.rectangle(color_image, (x, y), (x2, y2), color, 2)

            # display name
            class_name = object_detection.classes[class_id]
            cv2.putText(color_image, f"{class_name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Get center of the bbox
            cx, cy = (x + x2) // 2, (y + y2) // 2
            vi_tri = int(map_to_range(cx,width))
            distance = camera.get_distance_point(depth_image, cx, cy)

            # Draw circle
            cv2.circle(color_image, (cx, cy), 5, color, -1)
            cv2.putText(color_image, f"Distance: {distance} cm", (cx, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            if class_name == "KhungRo":
                cv2.putText(color_image, "lech tam bang: ", (0,height-100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.putText(color_image, f"{vi_tri}", (0,height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if class_name == "Banh":
                cv2.putText(color_image, "Tam bong: ", (width-230,height-100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.putText(color_image, f"(x,y): {cx,cy}", (width-230,height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        # daoanh = dao_anh(color_image)
        
        cv2.rectangle(color_image, (width//2,0), (width//2,height), (0,255,0), 2)
        # show color image
        cv2.imshow("Color Image", color_image)
        # cv2.imshow("depth Image", depth_image)
        key = cv2.waitKey(1)
        if key == 27:
            break

    # release the camera
    camera.release()
    cv2.destroyAllWindows()

def detect_ro_cam_0():
    cap = cv2.VideoCapture(0)
    while True:
        # Get color_image from realsense camera
        ret, color_image = cap.read()
        height, width, _ = color_image.shape
        # Get the object detection
        # getContours(color_image, draw=True, minArea=50000,filter=4)
        bboxes, class_ids, score = object_detection.detect(color_image,640, device='0',classes=[0,2])
        cv2.rectangle(color_image, (0,height-150), (250,height), (255,255,255), -1)
        cv2.rectangle(color_image, (width,height-150), (width-250,height), (255,255,255), -1)
        for bbox, class_id, score in zip(bboxes, class_ids, score):
            x, y, x2, y2 = bbox
            color = object_detection.colors[class_id]
            cv2.rectangle(color_image, (x, y), (x2, y2), color, 2)

            # display name
            class_name = object_detection.classes[class_id]
            cv2.putText(color_image, f"{class_name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Get center of the bbox
            cx, cy = (x + x2) // 2, (y + y2) // 2
            vi_tri = int(map_to_range(cx,width))

            if class_name == "KhungRo":
                cv2.putText(color_image, "lech tam bang: ", (0,height-100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.putText(color_image, f"{vi_tri}", (0,height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if class_name == "Banh":
                cv2.putText(color_image, "Tam bong: ", (width-230,height-100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.putText(color_image, f"(x,y): {cx,cy}", (width-230,height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        # daoanh = dao_anh(color_image)
        
        cv2.rectangle(color_image, (width//2,0), (width//2,height), (0,255,0), 2)
        # show color image
        cv2.imshow("Color Image", color_image)
        # cv2.imshow("depth Image", depth_image)
        key = cv2.waitKey(1)
        if key == 27:
            break

    # release the camera
    cap.release()    
    cv2.destroyAllWindows()


def detect_mau():

    while (1):
        ret, color_image, depth_image = camera.get_frame_stream()
        # Chuyển đổi khung hình sang không gian màu HSV
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Định nghĩa dải màu đỏ trong không gian màu HSV
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        # Kết hợp hai mặt nạ để phát hiện màu đỏ
        mask = mask1 + mask2

        # Áp dụng mask lên khung hình gốc
        red_output = cv2.bitwise_and(color_image, color_image, mask=mask)

        # Hiển thị kết quả
        cv2.imshow('Original', color_image)
        cv2.imshow('Red Detected', red_output)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # release the camera
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_ro_cam_0()
    # detect_ro()
    # detect_mau()