import cv2

video_path = '/Users/himanshuvyas/Downloads/vision-assistant/Video/streetwalking.mp4'

cap = cv2.VideoCapture(video_path) # create video capture object

while True:
    ret,frame = cap.read() # ret is boolean value to get if frame captured or not and frame is getting numpy values of video for each frame

    if not ret:
        break

    #Draw Rectangle
    cv2.rectangle(img=frame,pt1=(100,100),pt2=(400,300),color=(0,255,0),thickness=2)

    #Draw Text
    cv2.putText(img=frame,text="Street Video",org=(100,80),fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=1,color=(0,255,0),thickness=2)

    cv2.imshow("Street Video",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

