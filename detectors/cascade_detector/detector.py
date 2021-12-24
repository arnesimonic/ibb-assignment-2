import cv2, sys, os


class Detector:

	left_ear_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_leftear.xml'))
	right_ear_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_rightear.xml'))

	def detect(self, img):

		le_list = self.left_ear_cascade.detectMultiScale(img, 1.05, 1)
		re_list = self.right_ear_cascade.detectMultiScale(img, 1.05, 1)

		le_list_len = len(le_list)
		re_list_len = len(re_list)

		joint_list = []

		if le_list_len > 0:
			joint_list.extend(le_list)

		if re_list_len > 0:
			joint_list.extend(re_list)

		return joint_list

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	detector = Detector()
	detected_loc = detector.detect(img)
	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imwrite(fname + '.detected.jpg', img)