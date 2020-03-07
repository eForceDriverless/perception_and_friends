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

    # video
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter('output' + t + '.avi', fourcc, 20.0, (width, height))

        f = open('boxes1' + t + '.txt', 'w')
        


    img_i = 0
    key = ''
    while key != 113 and key != 27:  # for 'q' key
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(color)
            color_image = color.get_data()[:, :, :3]

            # color_image = cv2.imread("gfr_2019/00000.jpg")
            color_image_RGB = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)





            img_i += 1

            print(f"image: {img_i}")

            img = transforms.ToTensor()(color_image_RGB)
            # print(img.shape)
            # print(type(img))

            img, _ = pad_to_square(img, 0)

            # print(f"image type: {type(img)}")
            # print(f"image shape: {img.shape}")
            img = resize(img, img_size)
            img = img.unsqueeze(0)

            input_imgs = Variable(img.type(Tensor))
            # print(f"batch dim: {input_imgs.shape}")

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, conf_thresh, nms_thresh)
                # print(detections)

            # if detections is not None:
            # print(detections)
            # print(len(detections))
            if detections[0] != None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections[0], img_size, (color_image.shape[0], color_image.shape[1]))
                # unique_labels = detections[:, -1].cpu().unique()
                # n_cls_preds = len(unique_labels)
                # bbox_colors = random.sample(colors, n_cls_preds)
                # print(detections)

                # new frame
                f.write(f"{img_i}, {detections.shape[0]}\n")


                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                    # print(color_image.shape)
                    color_image = np.ascontiguousarray(color_image, dtype=np.uint8)

                    colors = [(255, 0, 0), (0, 255, 255), (0, 165, 255), (255, 255, 255)]
                    f.write(f"{x1}, {y1}, {x2}, {y2}, {classes[int(cls_pred)]}\n")
                    color_image = cv2.rectangle(color_image, (x1, y1), (x2, y2), colors[int(cls_pred)], 3)


                # cv2.imwrite("test.jpg", color_image)
                # 5 / 0


            if record:
                out.write(color_image)

            if display:
                cv2.imshow("ZED", color_image)
            key = cv2.waitKey(1)
            if img_i != 1:
                new_time = time.time()
                fps = 1 / (new_time - prev_time)
                font = cv2.FONT_HERSHEY_SIMPLEX
                color_image = np.ascontiguousarray(color_image, dtype=np.uint8)
                cv2.putText(color_image, f'FPS: {fps}', (10, 35), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print("FPS: ", fps)

            prev_time = time.time()
    cv2.imwrite("test.jpg", color_image)
    if record:
        out.release()
        f.close()
    cv2.destroyAllWindows()






