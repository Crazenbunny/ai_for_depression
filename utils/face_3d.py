# --------------------------------------------------------
# get 3d face key points
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

import face_alignment
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


start = time.time()
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True,face_detector='blazeface')
cap = cv2.VideoCapture('test/assets/test 00_00_00-00_00_18.mov')
svVideo=("output//test 00_00_00-00_00_18.avi")
fps = cap.get(cv2.CAP_PROP_FPS)
landmarks_3d = []
nonDetectFr = []
my_figsize, my_dpi = (20, 10), 80
totalIndx = 0
vis=0
fourcc = cv2.VideoWriter_fourcc(*'XVID') # create VideoWriter object
totalFrame = np.int32(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames: ", totalFrame)
width, height = my_figsize[0] * my_dpi, my_figsize[1] * my_dpi
out = cv2.VideoWriter(svVideo, fourcc, fps, (width, height))
while(cap.isOpened()):
    frameIndex = np.int32(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print("Processing frame ", frameIndex, "...")
    ret, frame = cap.read()
    if ret==True:
    # operations on the frame
        try:
            # generate face bounding box and track 2D landmarks for current frame
            frame_landmarks = fa.get_landmarks(frame)[-1]
        except:
            print("Landmarks in frame ", frameIndex, " (", frameIndex/fps, " s) could not be detected.")
            nonDetectFr.append(frameIndex/fps)
            continue

        landmarks_3d.append(frame_landmarks)
        totalIndx = totalIndx + 1
    else:
        break

    #################可视化################
    fig = plt.figure(figsize=my_figsize, dpi=my_dpi)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(6, 3)
    gs.update(wspace=0.5)
    gs.update(hspace=1)

    # ##################### Raw data with landmarks ######################
    im1 = frame  # with background
    ax1 = plt.subplot(gs[:6, :3])
    ax1.imshow(im1)
    plt.scatter(frame_landmarks[:,0], frame_landmarks[:,1], 2)
    ax1.set_title('Raw RGB Video', fontsize=32)
    ax1.set_ylabel('Pixel', fontsize=28)
    fig.canvas.draw()
    outFrame = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)

    if (vis):
        cv2.imshow('frame', outFrame)
    # write the flipped frame
    # out.write(outFrame)
    plt.close()  

cap.release()


end = time.time()
print("processing time:" + str(end - start))

np.save('3d_landmarks', np.asarray(landmarks_3d))

#实例视频用了8728s，共22435帧，大概2.5帧/s
