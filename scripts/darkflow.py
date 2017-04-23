#!/usr/bin/env python

from net.build import TFNet
import cv2
import rospy
from tlight_node import TLightNode
import argparse


def process(model, img):
    result = model.return_predict(img[None, :, :, :])
    return result


def get_model():
    options = {"model": "/home/chris/catkin_ws/src/yolo_light/scripts/cfg/tiny-yolo-udacity.cfg", "backup": "/home/chris/catkin_ws/src/yolo_light/scripts/ckpt/","load": 8987, "gpu": 1.0}
    model = TFNet(options)
    return model


def main():
    parser = argparse.ArgumentParser(description='Model Runner')
    args = parser.parse_args()
    node = TLightNode(lambda: get_model(), process)
    rospy.spin()


if __name__ == '__main__':
    main()
