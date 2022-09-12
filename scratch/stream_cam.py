import cv2 as cv
cap = cv.VideoCapture(1)  # make sure this number is correct for the computer
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    # print(frame.shape)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.resize(frame, (960, 540))  # resize because HDMI is pretty high resolution and viewing with cv is basic
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()