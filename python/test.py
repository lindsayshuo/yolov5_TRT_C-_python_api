import cv2
import TRTYolov5 as t


#
engine = t.create('../yolov5s.engine')

img = cv2.imread('./sendpix0.jpg')

b = t.detect(engine, img, 0.45)

#t.destroy(engine)

print(b)


# import ctypes
#
# # ctypes.CDLL("./build/libmyplugins.so")
# so = ctypes.CDLL("../build/libyolov5_trt.so")
# engine = so.create('../yolov5s.engine')
