import sys
import numpy as np
import cv2
import pyzed.sl as sl
import time

import torch
from models import *
from utils.utils import *
from utils.datasets import *

import torchvision.transforms as transforms

def get_xy_pos(cone_x, cone_y, camera_height, WIDTH=1280, HEIGHT=720):
    vert_angle = 90 - (60 + (60 - (cone_y / HEIGHT) * 60))
    vert_angle_rad = (vert_angle * math.pi) /180
    y_pos = camera_height / math.tan(vert_angle_rad)

    hor_angle = 90 - (45 + (cone_x / WIDTH) * 90)
    hor_angle_rad = (hor_angle * math.pi) / 180
    x_pos = math.tan(hor_angle_rad) * y_pos

    return x_pos, y_pos

def draw_field(frame, point1, point2, WIDTH=1280, HEIGHT=720):

    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), -1)

    return frame

if __name__ == '__main__':
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720
    init_params.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE
    init_params.camera_fps = 15
    init_params.sdk_verbose = True

    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()

    # args
    record = True
    display = True

    t = str(time.time())

    # model init
    yolo_architecture="config/yolov3-tiny-custom.cfg"
    img_size = 416
    weights_path = "weights/cones_5_epochs.pth"
    weights_path = "weights/cones_10_epochs.pth"
    weights_path = "weights/cones_15_epochs.pth"
    class_path = "data/coco.names"

    nms_thresh = 0.4
    conf_thresh = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(yolo_architecture, img_size=img_size).to(device)
    model.load_state_dict(torch.load(weights_path))
    classes = load_classes(class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    model.eval()

    # Get image size
    image_size = zed.get_resolution()
    width = image_size.width
    height = image_size.height
    print(width, height)

    runtime = sl.RuntimeParameters()
    color = sl.Mat()
    depth = sl.Mat()

    # homography params
    camera_height = 1.15

    WIDTH = 1280
    HEIGHT = 720

    # video
    font = cv2.FONT_HERSHEY_SIMPLEX
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter('videos/' + t + '.avi', fourcc, 20.0, (width, height))

        f = open('boxes/' + t + '.txt', 'w')

    img_i = 0
    key = ''
    frame_time = time.time()
    while key != 113 and key != 27:  # for 'q' key
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            img_i += 1
            zed.retrieve_image(color)
            color_image = color.get_data()[:, :, :3]

            # color_image = cv2.imread("gfr_2019/00000.jpg")
            color_image_RGB = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


            print(f"image: {img_i}")

            img = transforms.ToTensor()(color_image_RGB)

            img, _ = pad_to_square(img, 0)

            img = resize(img, img_size)
            img = img.unsqueeze(0)

            input_imgs = Variable(img.type(Tensor))

            if record:
                out.write(color_image)

            # draw bird-eye map
            field_meters = (7, 10)

            x1, y1 = (int(WIDTH / 100 * 80), int(HEIGHT / 100 * 40))
            x2, y2 = (int(WIDTH / 100 * 95), int(HEIGHT / 100 * 5))
            field_h = y1 - y2
            field_w = x2 - x1
            color_image = np.ascontiguousarray(color_image, dtype=np.uint8)
            color_image = draw_field(color_image, (x1, y1), (x2, y2))

            car_x = int((x1 + x2) / 2)
            car_y = int(y1 - field_h * 0.1)

            color_image = cv2.circle(color_image, (car_x, car_y), 5, (0, 255, 0), -1)

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, conf_thresh, nms_thresh)

            if detections[0] != None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections[0], img_size, (color_image.shape[0], color_image.shape[1]))

                # new frame
                if record:
                    f.write(f"{img_i}, {detections.shape[0]}\n")

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    colors = [(255, 0, 0), (0, 255, 255), (0, 165, 255), (255, 255, 255)]
                    color_bgr = colors[int(cls_pred)]

                    # cone mid point
                    cone_x, cone_y = (int((x1 + x2) / 2), int(y2))

                    x_pos, y_pos = get_xy_pos(cone_x, cone_y, camera_height)
                    dist = math.sqrt(x_pos**2 + y_pos**2)


                    # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                    color_image = np.ascontiguousarray(color_image, dtype=np.uint8)

                    f.write(f"{x1}, {y1}, {x2}, {y2}, {classes[int(cls_pred)]}\n")
                    color_image = cv2.rectangle(color_image, (x1, y1), (x2, y2), color_bgr, 3)
                    cv2.putText(color_image, "(%.2f, %.2f), dist: %.2f," % (x_pos, y_pos, dist), (cone_x, y1), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                    # cv2.putText(color_image, "conf: %.2f, dist: %.2f," % (cls_conf, dist), (cone_x, y1), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
                    # cv2.putText(color_image, "conf: %.2f, dist: %.2f," % (cls_conf, dist), (cone_x, y1), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

                    x_point = int(field_w * (x_pos / field_meters[0]))
                    y_point = int(field_h * (y_pos / field_meters[1]))

                    frame = cv2.circle(color_image, (car_x - x_point, car_y - y_point), 4, color_bgr, -1)
            elif record:
                f.write(f"{img_i}, 0\n")


            new_time = time.time()
            fps = int(1 / (new_time - frame_time))
            frame_time = new_time
            color_image = np.ascontiguousarray(color_image, dtype=np.uint8)
            cv2.putText(color_image, f'FPS: {fps}', (10, 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

            if display:
                cv2.imshow("ZED", color_image)
            key = cv2.waitKey(1)



            prev_time = time.time()
    cv2.imwrite("test.jpg", color_image)
    if record:
        out.release()
        f.close()
    cv2.destroyAllWindows()






