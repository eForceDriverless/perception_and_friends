import cv2
import math

def draw_field(frame, point1, point2, WIDTH=1280, HEIGHT=720):

    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), -1)

    return frame

def color_to_bgr(color):
    color_dict = {
            'orange': (0, 165, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'borange': (255, 255, 255)
            }

    return color_dict[color]

def get_xy_pos(cone_x, cone_y, camera_height, WIDTH=1280, HEIGHT=720):
    vert_angle = 90 - (60 + (60 - (cone_y / HEIGHT) * 60))
    vert_angle_rad = (vert_angle * math.pi) /180
    y_pos = camera_height / math.tan(vert_angle_rad)

    hor_angle = 90 - (45 + (cone_x / WIDTH) * 90)
    hor_angle_rad = (hor_angle * math.pi) / 180
    x_pos = math.tan(hor_angle_rad) * y_pos

    return x_pos, y_pos


# file name without extension
# name = 'first_tests/kuzely_rovina'
name = 'first_tests/finalni_zatacka'

cap = cv2.VideoCapture(name + '.avi')
font = cv2.FONT_HERSHEY_SIMPLEX
f = open(name + '.txt', 'r')

# # recording
# fourcc = cv2.VideoWriter_fourcc(*'X264')
# out = cv2.VideoWriter('videos/' + "birdview_" + name + '.avi', fourcc, 20.0, (1280, 720))

camera_height = 1.2

WIDTH = 1280
HEIGHT = 720

prev_frame_val = True
frame_idx = 0
while cap.isOpened():

    ret, frame = cap.read()

    meta_line = f.readline()
    orig_idx, box_count = [int(word.strip()) for word in meta_line.split(',')]
    # orig_idx, box_count = meta_line.split(',')
    print("orig_idx:", orig_idx)
    print("box_count:", box_count)

    if ret == True:

        print("frame_shape:", frame.shape)

        field_meters = (7, 10)

        x1, y1 = (int(WIDTH / 100 * 80), int(HEIGHT / 100 * 40))
        x2, y2 = (int(WIDTH / 100 * 95), int(HEIGHT / 100 * 5))
        field_h = y1 - y2
        field_w = x2 - x1
        frame = draw_field(frame, (x1, y1), (x2, y2))

        car_x = int((x1 + x2) / 2)
        car_y = int(y1 - field_h * 0.1)

        frame = cv2.circle(frame, (car_x, car_y), 5, (0, 255, 0), -1)

        min_angle = 200
        max_angle = 0
        for i in range(box_count):
            box = f.readline()
            x1, y1, x2, y2, color = [val.strip() for val in box.split(',')]
            x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))

            color_bgr = color_to_bgr(color)
            # frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr , 5)

            cone_x, cone_y = (int((x1 + x2) / 2), int(y2))

            x_pos, y_pos = get_xy_pos(cone_x, cone_y, camera_height)
            dist = math.sqrt(x_pos**2 + y_pos**2)

            cv2.putText(frame, "(%.2f, %.2f), dist: %.2f" % (x_pos, y_pos, dist), (cone_x, y1), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

            x_point = int(field_w * (x_pos / field_meters[0]))
            y_point = int(field_h * (y_pos / field_meters[1]))

            frame = cv2.circle(frame, (car_x - x_point, car_y - y_point), 4, color_bgr, -1)
            frame = cv2.circle(frame, (cone_x, cone_y), 2, (0, 0, 255), -1)
            print(type(x1))

        print("max_angle:", max_angle)
        print("min_angle:", min_angle)

        print(frame.shape)
        # out.write(frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    else:
        break

    # line = f.readline()
    # print(line)
    a = input()

cap.release()
cv2.destroyAllWindows()
out.release()
