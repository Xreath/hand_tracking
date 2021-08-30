#sol ele ve sağ ele göre ayrı olarak ayarlama yapılacak


import cv2
import mediapipe as mp
import time
import math
import imutils


def gradient_calculate(p1, p2):
    gradient = (p2[1]-p1[1])/(p2[0]-p1[0]+0.00001)
    return gradient

def get_angle(p1, p2, p3):
    m1 = gradient_calculate(p1, p2)
    m2 = gradient_calculate(p1, p3)
    angR = math.atan((m2-m1)/(1+(m2*m1)))
    angR = round(math.degrees(angR))
    angR = -(angR-7)
    if(angR<0):
        angR=180+angR
        return angR
    return  angR


def which_hand(results,):
    label_array=[]

    if results.multi_handedness:
        for handtype in results.multi_handedness:
            for id,tp in enumerate(handtype.classification):
                print(tp.label)
                label_array.append(tp.label)
    return label_array

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDrawing=mp.solutions.drawing_utils

pTime=0
cTime=0
angle=0
cap=cv2.VideoCapture(0)

while True:

    success,image=cap.read()
    image_height, image_width, _ = image.shape
    imgRGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)

    label_array_hand=which_hand(results)

    if results.multi_hand_landmarks:
        for handMHL in results.multi_hand_landmarks:
            landmark_dict={}
            for id,lm in enumerate(handMHL.landmark):
                cx,cy=int(lm.x*image_width),int(lm.y*image_height)
                label_array_hand = which_hand(results)
                landmark_dict[str(id)] =(cx,cy)
                x, y = landmark_dict["0"]
                if(id>13):
                    if(label_array_hand[0]=="Right"):
                        cv2.line(image, landmark_dict["0"], landmark_dict["9"], (255, 0, 0), 4)
                       # cv2.line(image, (0, landmark_dict["0"][1]), (image_width, landmark_dict["0"][1]),(255, 255, 255), 2)
                        angle = get_angle(landmark_dict["0"], landmark_dict["9"], (0, landmark_dict["0"][1]))

                        cv2.putText(image, "Angle: " + str(angle), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)
            mpDrawing.draw_landmarks(image,handMHL,mpHands.HAND_CONNECTIONS)  

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(image,str(int(fps)),(10,70),cv2.FONT_ITALIC,1,(255 ,255 ,0),1)

    image = imutils.rotate(image, (int(angle)-90))



    cv2.imshow("Image",image)
    k=cv2.waitKey(1)
    
    if(k==27):
        break
    
cap.release()
cv2.destroyAllWindows()


