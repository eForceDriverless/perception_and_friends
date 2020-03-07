import cv2

name = 'kuzely_rovina'

cap = cv2.VideoCapture(name + '.avi')
f = open(name + '.txt', 'r')

frame_idx = 0
while cap.isOpened():

    ret, frame = cap.read()
    if ret == True:
        print(frame.shape)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    else:
        break

    line = f.readline()
    print(line)
    a = input()

cap.release()
cv2.destroyAllWindows()
