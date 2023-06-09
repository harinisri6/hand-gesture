omport='com4'
import numpy as np
import cv2
from tensorflow  import keras
from tensorflow .keras.preprocessing.image import ImageDataGenerator
from keras import backend as K




model = keras.models.load_model("./model_mob.h5",compile=False)
import time
import serial
background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

word_dict = {0:'no',1:'hungry',2:'medicine',3:'water',4:'good luck',5:'walk',6:'Sorry',7:'Sleep',8:'washroom',9:'emergency'}
pred=0
cnt=0
count=0
laststate=0
def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)



def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        return (thresholded, hand_segment_max_cont)

cam = cv2.VideoCapture(0)
amount_of_frames = cam.get(cv2.CAP_PROP_FPS)
print(amount_of_frames)
num_frames =0
while True:
    if(count!=15):
        ret, frame = cam.read()
        count=count+1
        continue
    else:
        count=0
        ret, frame = cam.read()
    
        frame = cv2.flip(frame, 1)
    
        frame_copy = frame.copy()
    
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
    
    
        if num_frames < 70:
            
            cal_accum_avg(gray_frame, accumulated_weight)
            
            cv2.putText(frame_copy, "TAKING BACKGROUND", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
        else: 
            # segmenting the hand region
            hand = segment_hand(gray_frame)
            
    
            if hand is not None:
                
                thresholded, hand_segment = hand
    
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)
                
                cv2.imshow("Thesholded Hand Image", thresholded)
                
                thresholded = cv2.resize(thresholded, (120,120))
                thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))
                cv2.putText(frame_copy, word_dict[np.argmax(pred)]+str(np.max(pred)), (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                
                pred = model.predict(thresholded,verbose=2)
            
        

        # Draw ROI on frame_copy
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)
    
        num_frames += 1
        predict=np.argmax(pred)
    
        cv2.imshow("Sign Detection", frame_copy)
        if(laststate!=predict):
            cnt=1
        else:
            cnt=0
        if(cnt==1 and pred.max()>0.4 ):
            print(pred.max(), word_dict[np.argmax(pred)])

            var="*"+str(predict)
            hardser=serial.Serial(comport,9600)
            for i in range(3):
                hardser.write(var.encode())  
            time.sleep(0.5)
            hardser.close()
            time.sleep(0.5)
        
            laststate=predict

            
        else:
            pass
    
    
        # Close windows with Esc
        k = cv2.waitKey(1) & 0xFF
    
        if k == 27:
            break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()
