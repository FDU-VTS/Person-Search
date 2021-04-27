import cv2


cap = cv2.VideoCapture("rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1")
ret, frame = cap.read()
while True:
    ret, frame = cap.read()
    if frame is None:
        print("Missing frame")
        cap = cv2.VideoCapture("rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1")
        continue
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
