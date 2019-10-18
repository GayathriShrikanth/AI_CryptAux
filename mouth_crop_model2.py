import cmake
import matplotlib.pyplot as plt
import dlib
import numpy as np
import cv2
import argparse
import os
import math
import skvideo
skvideo.setFFmpegPath("C:\\ffmpeg\\bin")
import skvideo.io
from PIL import Image

#pre-trained model for face recognition required by dlib
predictor_path = 'C:/Users/gayat/Downloads/shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#video_path='C:/Users/gayat/OneDrive/Pictures/Camera Roll/sample_video.mp4'
video_path='C:/gayathri/apple.mp4'

#Reads frames using FFmpeg
#skvideo.setFFmpegPath('C:/Users/gayat/PycharmProjects/untitled2/venv/Lib/site-packages/skvideo/io')
inputparameters = {}
outputparameters = {}
v = skvideo.io.vread(video_path)
reader = skvideo.io.FFmpegReader(video_path,inputdict=inputparameters,outputdict=outputparameters)

#print(skvideo.__file__)
print(type(reader))

(num_frames, h, w, c) = reader.getShape()
print(num_frames, h, w, c)
video_shape = reader.getShape()


# The required parameters
activation = []
max_counter = 150
total_num_frames = int(video_shape[0])
num_frames = min(total_num_frames,max_counter)
counter = 0
writer = skvideo.io.FFmpegWriter(video_path)

# Required parameters for mouth extraction.
width_crop_max = 0
height_crop_max = 0

#Check by displaying the frames
# img=Image.fromarray(v[1])
# img.save('sample.png')
#img.show()
plt.imshow(v[20])
plt.show()

# Loop over all frames.
for frame in v:
    print('frame_shape:', frame.shape)

    # Process the video and extract the frames up to a certain number and then stop processing.
    if counter > num_frames:
        break

    # Detection of the frame
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detections = detector(frame, 1)

    # 20 mark for mouth
    marks = np.zeros((2, 20))

    # All unnormalized face features.
    Features_Abnormal = np.zeros((190, 1))

    # Number of faces detected detected.
    print(len(detections))
    if len(detections) > 0:
        for k, d in enumerate(detections):

            # Shape of the face.
            # This object is a tool that takes in an image region
            # containing some object and outputs a set of point locations that define the pose of the object.

            shape = predictor(frame, d)
            co = 0
            # Specific for the mouth.
            for n in range(48, 68):
                """
                This for loop is going over all mouth-related features.
                X and Y coordinates are extracted and stored separately.
                """
                X = shape.part(n)
                A = (X.x, X.y)
                marks[0, co] = X.x
                marks[1, co] = X.y
                co += 1

            # Get the extreme points(top-left & bottom-right)
            X_left, Y_left, X_right, Y_right = [int(np.amin(marks, axis=1)[0]), int(np.amin(marks, axis=1)[1]),int(np.amax(marks, axis=1)[0]),int(np.amax(marks, axis=1)[1])]

            # Find the center of the mouth.
            X_center = (X_left + X_right) / 2.0
            Y_center = (Y_left + Y_right) / 2.0

            # Make a boarder for cropping.
            border = 30
            X_left_new = X_left - border
            Y_left_new = Y_left - border
            X_right_new = X_right + border
            Y_right_new = Y_right + border

            # Width and height for cropping(before and after considering the border).
            width_new = X_right_new - X_left_new
            height_new = Y_right_new - Y_left_new
            width_current = X_right - X_left
            height_current = Y_right - Y_left

            # Determine the cropping rectangle dimensions(the main purpose is to have a fixed area).
            if width_crop_max == 0 and height_crop_max == 0:
                width_crop_max = width_new
                height_crop_max = height_new
            else:
                width_crop_max += 1.5 * np.maximum(width_current - width_crop_max, 0)
                height_crop_max += 1.5 * np.maximum(height_current - height_crop_max, 0)

            X_left_crop = int(X_center - width_crop_max / 2.0)
            X_right_crop = int(X_center + width_crop_max / 2.0)
            Y_left_crop = int(Y_center - height_crop_max / 2.0)
            Y_right_crop = int(Y_center + height_crop_max / 2.0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            if X_left_crop >= 0 and Y_left_crop >= 0 and X_right_crop < w and Y_right_crop < h:
                mouth = frame[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop, :]

                # Save the mouth area.
                mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)

                mouth_destination_path="C:/gayathri/imag2"
                cv2.imwrite(mouth_destination_path + '/' + 'frame' + '_' + str(counter) + '.png', mouth_gray)

                print("The cropped mouth is detected ...")
                activation.append(1)
            else:
                cv2.putText(frame, 'The full mouth is not detectable. ', (30, 30), font, 1, (0, 255, 255), 2)
                print("The full mouth is not detectable. ...")
                activation.append(0)

    else:
        cv2.putText(frame, 'Mouth is not detectable. ', (30, 30), font, 1, (0, 0, 255), 2)
        print("Mouth is not detectable. ...")
        activation.append(0)

    if activation[counter] == 1:
        # Demonstration of face.
        cv2.rectangle(frame, (X_left_crop, Y_left_crop), (X_right_crop, Y_right_crop), (0, 255, 0), 2)
        # cv2.imshow('frame', frame)
    print('frame number %d of %d' % (counter, num_frames))

    # write the output frame to file
    print("writing frame %d with activation %d" % (counter + 1, activation[counter]))
    writer.writeFrame(frame)
    counter += 1

writer.close()
