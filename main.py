import string
import numpy as np
import cv2
import win32gui
import win32ui
from PIL import Image
from ctypes import windll
import time
from config import *
from pynput import mouse
from pynput.mouse import Button

i = 0

mouse = mouse.Controller()

hwnd = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive - Direct3D 9')
left, top, right, bot = win32gui.GetWindowRect(hwnd)
w = right - left
h = bot - top

hwndDC = win32gui.GetWindowDC(hwnd)
mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
saveDC = mfcDC.CreateCompatibleDC()

net = cv2.dnn.readNet("yolov4_tiny_custom_last.weights", "yolov4_tiny_custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

labels = open("obj.names").read().strip().split("\n")

font_scale = 2
thickness = 4

colors = [[255,255,255],[0,0,255],[255,0,0],[0,255,255]]
print(colors)

def yolo(img):
    global boxs, width, height
    boxs = []
    img = cv2.resize(np.array(img), None, fx=1, fy=1)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(output_layers)

    global boxes, idxs, class_ids, scores, class_ids_exit
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                #print(boxes)
    #font = cv2.FONT_HERSHEY_PLAIN
# ensure at least one detection exists
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    class_ids_exit = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            class_ids_exit.append(class_ids[i])
            boxs.append([x,y,w,h])
            #boxs.append([x,y,w,h])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            # calculate text width & height to draw the transparent boxes as background of the text
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
            cv2.circle(img,(round(x + w/2), round(y + h/8)), 2, [255,255,255], 2)

    return img



def scrnshot():
    global saveDC

    # screen capture
    #img = ImageGrab.grab(bbox=(0,0,1680,1080))
    #img_np = np.array(img)
    #frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    #cv2.imwrite('frame.jpg', frame)
    global saveBitMap
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    # Change the line below depending on whether you want the whole window
    # or just the client area. 
    #result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    img = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)


    if result == 1:
        return img


screenshotting = bool(input('Screenshotting(bool):'))
dtime_last = time.perf_counter()
while True:
    try:
        i +=1
        img = np.array(scrnshot())
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('image',cv2.resize(yolo(img), None, fx=0.4, fy=0.4))
        cv2.waitKey(1)
        print(class_ids)
        print(boxs)
        if time.perf_counter() - dtime_last >= 1 and screenshotting == True and len(class_ids_exit) !=0:
            c=time.perf_counter()
            cv2.imwrite(f'generated_data/{c}.jpg',img)
            dtime_last = time.perf_counter()
            with open(f'generated_data/{c}.txt','w+') as file:
                n = 0
                for i in class_ids_exit:
                    p = f'{boxs[n][0]/width+boxs[n][2]/width/2} {boxs[n][1]/height+boxs[n][3]/height/2} {boxs[n][2]/width} {boxs[n][3]/height}'
                    file.write(f'{i} {p}\n')
                    n+=1


        win32gui.DeleteObject(saveBitMap.GetHandle())
    except KeyboardInterrupt:
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        cv2.destroyAllWindows()
        exit()
    except NameError:
        print('bad')
        pass