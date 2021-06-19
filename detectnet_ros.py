"""
jetson nano object detection 
https://rawgit.com/dusty-nv/jetson-inference/dev/docs/html/python/jetson.inference.html#detectNet
"""

import jetson.inference
import jetson.utils
import rospy
import cv2
import numpy as np
import subprocess, shlex, psutil
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

from darknet_ros_msgs.msg import BBoxes, BBox

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(1280,720,"/dev/video0")    # '/dev/video0' for V4L2
compmsg = CompressedImage()
rospy.init_node("visualizer")


def detection_and_publish(data) :
	
	while True :
		img,width,height = camera.CaptureRGBA()
		detections = net.Detect(img)

		bboxes = BBoxes()
		bboxes.header.stamp = rospy.Time.now()
		bboxes.header.frame_id = "detection"

		for list in detections :
			bbox = BBox()
			bbox.probability = list.Confidence
			bbox.cx = list.Center.x
			bbox.cy = list.Center.y
			bbox.area = list.Area
			bbox.id = list.ClassID
			bboxes.bboxes.append(bbox)

		#convert to ros 
		numpyImg = jetson.utils.cudaToNumpy(img, width, height, 4)
		aimg1 = cv2.cvtColor(numpyImg.astype(np.uint8), cv2.COLOR_RGB2BGR)
		compmsg.header.stamp = rospy.Time.now()
		compmsg.format = "jpeg"
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
		compmsg.data = np.array(cv2.imencode('.jpg', aimg1, encode_param)[1]).tostring()
		comp_img_pub.publish(compmsg)
		bboxes_pub.publish(bboxes)


detection_and_publish()
comp_img_pub = rospy.Publisher("/camera_nano/object_detect/image_raw/compressed", CompressedImage, queue_size = 1)
bboxes_pub = rospy.Publisher("/nano_detection/bboxes", BBoxes, queue_size = 1)
rospy.spin()

