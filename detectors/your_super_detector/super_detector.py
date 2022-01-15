import cv2, sys, os
import numpy as np


class SuperDetector:

	net = cv2.dnn.readNet(
		"C:\\Users\\arnes\\Faks Local\\SB\\DN2\\repo\\detectors\\your_super_detector\\yolov3_training_last.weights",
		"C:\\Users\\arnes\\Faks Local\\SB\\DN2\\repo\\detectors\\your_super_detector\\yolov3_testing.cfg"
	)
	layer_names = net.getLayerNames()

	def detect(self, img):
		output_layers = [self.layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]

		height, width, channels = img.shape

		# Detecting objects
		blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

		self.net.setInput(blob)
		outs = self.net.forward(output_layers)

		boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.3:
					# Object detected
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					# Rectangle coordinates
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)

					boxes.append([x, y, w, h])

		return boxes

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	detector = SuperDetector()
	detected_loc = detector.detect(img)
	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imwrite(fname + '.detected.jpg', img)