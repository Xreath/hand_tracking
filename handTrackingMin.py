import cv2
import mediapipe as mp
import time

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDrawing=mp.solutions.drawing_utils
id_cordinate={}
pTime=0
cTime=0

cap=cv2.VideoCapture(0)

while True:
    
    success,image=cap.read()
    image_height, image_width, _ = image.shape
    imgRGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handMHL in results.multi_hand_landmarks:
            print(handMHL)
            for id,lm in enumerate(handMHL.landmark):
                id_cordinate[str(id)]=str(lm)
            mpDrawing.draw_landmarks(image,handMHL,mpHands.HAND_CONNECTIONS)


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(image,str(int(fps)),(10,70),cv2.FONT_ITALIC,1,(255 ,255 ,0),1)

    cv2.imshow("Image",image)
    k=cv2.waitKey(1)
    
    if(k==27):
        break
    
cap.release()
cv2.destroyAllWindows()