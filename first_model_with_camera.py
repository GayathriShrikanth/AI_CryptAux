from imutils.video import FPS
import argparse
import os
import imutils
import cv2
from keras.applications.vgg16 import VGG16
from scipy import misc # run "pip install pillow"
from imutils import face_utils
import dlib
import skvideo
import glob
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import pandas
skvideo.setFFmpegPath("C:\\ffmpeg\\bin")

predictor_path = 'C:/Users/gayat/Downloads/shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

RECTANGLE_LENGTH = 90

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=20,
				help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
				help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

print("[INFO] sampling frames from webcam...")
stream = cv2.VideoCapture(0)
fps = FPS().start()
i=0
# loop over some frames
while fps._numFrames < args["num_frames"]:
	# grab the frame from the stream and resize it to have a maximum
	# width of 400 pixels
	(grabbed, frame) = stream.read()

	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)
	rect = rects[0]
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	(x, y, w, h) = face_utils.rect_to_bb(rect)
	w = RECTANGLE_LENGTH
	h = RECTANGLE_LENGTH
	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	(x_r, y_r, w_r, h_r) = (x, y, w, h)

	# show the face number
	cv2.putText(frame, "Face #{}".format(0 + 1), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	crop_img = frame[y_r:y_r + h_r, x_r:x_r + w_r]

	img_path='img'+str(i)+'.jpg'
	i+=1
	img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
	cv2.imwrite('camera/' + img_path, img_gray)

	# check to see if the frame should be displayed to our screen
	#if args["display"] > 0:
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()
sequence=[]
for file in glob.glob("camera/*.jpg"):
	basename = os.path.basename(file)
	image = misc.imread(file)
	print(image.shape)
	#image = np.reshape(image, 90 * 90 * 3)
	sequence.append(image)

samples = np.shape(sequence)[0]
h_size = 90
w_size = 90
chanel = 3
np.shape(sequence)
bottleneck_features = np.empty([samples,2048])
model = VGG16(weights='imagenet', include_top=False)

for j in range(len(sequence)):
    print("Image Number: ", j)
    img = np.expand_dims(sequence[j], axis=0)
    feature = model.predict(img)
    bottleneck_features[j] = feature.flatten()

np.save('camtest',bottleneck_features)
camera_test = np.load('camtest.npy')
model2= load_model('Dense_86.h5')
prediction = model2.predict(camera_test)
prediction_result = np.argmax(prediction, axis=1)
class_names={0:"Begin", 1:"Choose", 2:"Connection", 3:"Navigation", 4:"Next", 5:"Previous", 6:"Start", 7:"Stop", 8:"Hello", 9:"Web"}
for s in prediction_result:
	print(class_names[s])
