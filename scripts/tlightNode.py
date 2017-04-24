#!/usr/bin/env python

import threading
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class TLightNode(object):
    def __init__(self, get_model_callback, model_callback):
        rospy.init_node('tlightModel')
        self.model = get_model_callback()
        self.get_model = get_model_callback
        self.predict = model_callback
        self.bridge = CvBridge()
        self.count = 0
        self.boxes = None
        self.img =  None
        self.img_out = None
        self.image_lock = threading.RLock()
        #self.sub = rospy.Subscriber('/left_camera/image_color/compressed', CompressedImage, self.update_image)
        self.sub = rospy.Subscriber('/image_raw', Image, self.update_image)
        self.pub = rospy.Publisher('/out_image', Image, queue_size=1)
        rospy.Timer(rospy.Duration(0.5), self.callback_image)

    def update_image(self, img):
        arr = self.bridge.imgmsg_to_cv2(img,"bgr8") 
        #np_arr = np.fromstring(img.data, np.uint8)
        #arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #print(img.height,img.width) #raw image is of size (600,800)
        if self.image_lock.acquire(True):
            self.img = arr
            if self.model is None:
                self.model = self.get_model()
            self.img_out, self.boxes = self.predict(self.model, self.img)
            self.img_out = np.asarray(self.img_out[0,:,:,:])
            for box in self.boxes:
                if 'traffic light' in box['label']:
                    cv2.rectangle(self.img_out,(box['topleft']['x'], 
                                                box['topleft']['y']), 
                                                (box['bottomright']['x'], 
                                                box['bottomright']['y']), 
                                                (255,0,0), 6)
                    cv2.putText(self.img_out, box['label'], 
                               (box['topleft']['x'], 
                               box['topleft']['y'] - 12), 0, 0.6, (255,0,0) ,6//3)

            print(self.img_out.shape)
            self.image_lock.release()

    def callback_image(self, event):
        if self.img_out is None:
            return
        self.pub.publish(self.bridge.cv2_to_imgmsg(self.img_out, "bgr8"))
