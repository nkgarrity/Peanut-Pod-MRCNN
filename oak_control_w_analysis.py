# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:41:12 2022

@author: nkgarrit
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:31:48 2022

@author: nkgarrit
"""

import depthai as dai
import cv2
import numpy as np
from tkinter import *
from plantcv import plantcv as pcv
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import statistics
import gc
import sys
import random
import math
import skimage.io
import matplotlib
from PIL import Image

#ENSURE YOU ARE IN A DIRECTORY CONTAINING MRCNN FILES

# Root directory of the project
ROOT_DIR = (".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

# Step size ('W','A','S','D' controls)
STEP_SIZE = 8
# Manual exposure/focus/white-balance set step
EXP_STEP = 500  # us
ISO_STEP = 50
LENS_STEP = 3
WB_STEP = 200



# outputs = pd.DataFrame(columns = ["mean_length", "sd_length", "mean_width",
#  "sd_width", "h_l", "% jumbo", "% fancy", "% No1", "obj_num"])
# outputs.to_csv("outs.csv")





def findArucoMarkers(img, markerSize = 4, totalMarkers = 50, draw = True):
    global pixelsPerMetric
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = cv2.aruco.Dictionary_get(key)
    arucoParam = cv2.aruco.DetectorParameters_create()
    bboxs, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    arucofound =  bboxs, ids, rejected
    aruco_perimeter= cv2.arcLength(arucofound[0][0][0], True)
    pixelsPerMetric = aruco_perimeter / 200
    print("pixel to mm", pixelsPerMetric)
    gc.collect()



def analyze(file):

    
    global imgpcv, plants1, img1, image
    
    config = InferenceConfig()
    config.display()


    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights("./mask_rcnn_9_20.h5", by_name=True)
    class_names = ['BG','Peanut']

    

    image = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
    
    results = model.detect([image], verbose = 1)

    r = results[0]
       
    #commenting this out but leaving in for debugging purposes
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

    masks = r['masks']
    f,g,h = masks.shape
       
    img1 = np.zeros(shape=[3040, 4032], dtype=bool)
    for i in range(0,h):
        img1 = img1 | masks[:,:,i]
        

    img1 = np.array(img1, dtype=np.uint8)
    
    pcv.params.debug = None

    
    imgpcv = file
    

    objects, obj_hierarchy = pcv.find_objects(img=imgpcv, mask=img1)
    # # obj, mask = pcv.object_composition(img = img, contours = objects, hierarchy = obj_hierarchy)

    obj_num=len(objects)

    plotting_img = pcv.visualize.obj_sizes(img=imgpcv, mask = img1, num_objects = obj_num)

    shape_img = np.copy(imgpcv)



    for i in range(0,obj_num):
        if obj_hierarchy[0][i][3] == -1:
            seed, seed_mask = pcv.object_composition(img=imgpcv, contours=[objects[i]], hierarchy=np.array([[obj_hierarchy[0][i]]]))
            shape_img = pcv.analyze_object(img=shape_img, obj=seed, mask=seed_mask, label=f"seed{i}")
            color_img = pcv.analyze_color(rgb_img=imgpcv, mask=img1, hist_plot_type=None, label='default')


    pcv.plot_image(img=shape_img)

    plants1 = pcv.outputs.observations   
    
    el_maj = []
    el_min = [] 
    all_seeds = []        
      


    color_v = plants1["default"]["lightness_frequencies"]["value"]
    color_l = plants1["default"]["lightness_frequencies"]["label"]
    color_df = pd.DataFrame(list(zip(color_v, color_l)), columns = ["Value", "Label"])


    #Need a better way to do this but this removes the "default" key which does not have ellipse info just overall color data    

    del plants1["default"]
    
    all_seeds = list(plants1.keys())
        
    for l in all_seeds:
        if plants1[l]["area"]["value"] < 2000:
            del plants1[l]
        else:
            pass
    all_seeds = list(plants1.keys())    
   
    for j in all_seeds:
        # print(plants1[j]["ellipse_minor_axis"]["value"])
        el_min.append(plants1[j]["ellipse_minor_axis"]["value"])

    for h in all_seeds:
        # print(plants1[j]["ellipse_major_axis"]["value"])
        el_maj.append(plants1[h]["ellipse_major_axis"]["value"])

    df = pd.DataFrame(list(zip(all_seeds, el_maj, el_min)), columns = ["ID", "Major", "Minor"])

    df['Major'] = (df['Major'] / pixelsPerMetric)
    df['Minor'] = (df['Minor'] / pixelsPerMetric)
    

    
    df.to_csv("./raw_data/"+nm+".csv")
    count = len(df.axes[0])

    
    jumbo = sum((df.Minor > 14.68))
    fancy = sum((df.Minor >= 12.7) & (df.Minor <= 14.68))
    No1 = sum(df.Minor < 12.7)
    
    
    
    
    out1 = pd.DataFrame(
        {"mean_length" : [(sum(df.Major) / count)],
         "sd_length" : [statistics.stdev(df.Major)],
         "mean_width" : [(sum(df.Minor) / count)],
         "sd_width" : [statistics.stdev(df.Minor)],
         "h_l" : [sum(color_df["Value"] * color_df["Label"]) / 100],
         "% jumbo" : [jumbo / count],
         "% fancy" : [fancy / count],
         "% No1" : [No1 / count],
         "obj_num" : [count]},
        index=[nm])
    #dfap(out1)
    print(out1)
    out1.to_csv("outs.csv", mode = "a", index=True, header=False)
    


    pcv.outputs.clear()  
    

    
    gc.collect()
        
    

def get_entries():
    global y_entry, l_entry, p_entry
    y_entry = year.get("1.0", "end-1c")
    l_entry = location.get("1.0", "end-1c")
    p_entry = plot.get("1.0", "end-1c")

        
def clicked():
    root.destroy()

def name_box():
    global year, location, plot, root
    root = Tk()
    root.title("Name Changer")
    root.geometry('350x200')
    lblyear = Label(root, text = "Year")
    lblyear.grid()
    lbllocation = Label(root, text = "Location")
    lbllocation.grid(row=1)
    lblplot = Label(root, text = "Plot")
    lblplot.grid(row=2)
    year = Text(root, height = 1, width = 10)
    year.grid(column = 1, row = 0)
    location = Text(root, height = 1, width = 10)
    location.grid(column =1, row =1)
    plot = Text(root, height = 1, width = 10)
    plot.grid(column = 1, row = 2) 
    btn = Button(root, text = "Save" , fg = "red", command=get_entries)
    btn.grid(column=1, row=3)    
    btn = Button(root, text = "Done" , fg = "red", command=clicked)
    btn.grid(column=1, row=4)
    root.mainloop()

def clamp(num, v0, v1):
    return max(v0, min(num, v1))


def col_cor(source):
    global corrected_img
   
    
    pcv.params.debug = None
    
    target_img, t_path, t_filename = pcv.readimage(filename = "./ref_imgs/ref_color_3500.jpg")
    source_img, s_path, s_filename = pcv.readimage(filename = source)
    
    dataframe1, start, space = pcv.transform.find_color_card(rgb_img = target_img, background = "dark")
    
    dataframe2, start2, space2 = pcv.transform.find_color_card(rgb_img = source_img, background = "dark")
    
    target_mask = pcv.transform.create_color_card_mask(target_img, radius = 15,
                                                       start_coord = start, spacing = space, nrows = 6, ncols = 4)
    
    source_mask = pcv.transform.create_color_card_mask(source_img, radius = 15,
                                                       start_coord = start2, spacing = space2, nrows = 6, ncols = 4)
    
    tm, sm, transformation_matrix, corrected_img = pcv.transform.correct_color(target_img = target_img,
                                                                               target_mask = target_mask,
                                                                               source_img = source_img,
                                                                               source_mask = source_mask,
                                                                               output_directory = ".")
    corrected_img = np.array(corrected_img)
    return corrected_img


pipeline = dai.Pipeline()


#Sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
videoEncoder = pipeline.create(dai.node.VideoEncoder)
stillEncoder = pipeline.create(dai.node.VideoEncoder)

controlIn = pipeline.create(dai.node.XLinkIn)
configIn = pipeline.create(dai.node.XLinkIn)
videoMjpegOut = pipeline.create(dai.node.XLinkOut)
stillMjpegOut = pipeline.create(dai.node.XLinkOut)
previewOut = pipeline.create(dai.node.XLinkOut)


controlIn.setStreamName('control')
configIn.setStreamName('config')
videoMjpegOut.setStreamName('video')
stillMjpegOut.setStreamName('still')
previewOut.setStreamName('preview')

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
camRgb.setVideoSize(640, 360)
camRgb.setPreviewSize(300,300)

camRgb.setStillSize(4032, 3040)
videoEncoder.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
stillEncoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)


camRgb.video.link(videoEncoder.input)
camRgb.still.link(stillEncoder.input)
camRgb.preview.link(previewOut.input)
controlIn.out.link(camRgb.inputControl)
configIn.out.link(camRgb.inputConfig)
videoEncoder.bitstream.link(videoMjpegOut.input)
stillEncoder.bitstream.link(stillMjpegOut.input)

with dai.Device(pipeline) as device:
    
    #data queues
    controlQueue = device.getInputQueue('control')
    configQueue = device.getInputQueue('config')
    previewQueue = device.getOutputQueue('preview')
    videoQueue = device.getOutputQueue('video')
    stillQueue = device.getOutputQueue('still')
    
    expTime = 5000
    expMin = 1
    expMax = 33000
    
    sensIso = 200
    sensMin = 100
    sensMax = 1600
    
    while True:
        previewFrames = previewQueue.tryGetAll()
        for previewFrame in previewFrames:
            cv2.imshow('preview', previewFrame.getData().reshape(previewFrame.getHeight(), previewFrame.getWidth(), 3))


        videoFrames = videoQueue.tryGetAll()
        for videoFrame in videoFrames:
            # Decode JPEG
            frame = cv2.imdecode(videoFrame.getData(), cv2.IMREAD_UNCHANGED)
            # Display
            cv2.imshow('video', frame)
           
           
        stillFrames = stillQueue.tryGetAll()
        for stillFrame in stillFrames:
            # Decode JPEG
            frame = cv2.imdecode(stillFrame.getData(), cv2.IMREAD_UNCHANGED)
            # Display
            # cv2.imshow('still', frame)
            name_box()
            nm = y_entry+"-"+l_entry+"-"+p_entry
            nm1 = y_entry+"-"+l_entry+"-"+p_entry+".jpg"
            print("Image saved as "+y_entry+"-"+l_entry+"-"+p_entry+".jpg")
            cv2.imwrite("Test/"+nm1, frame)
            col_cor("Test/"+nm1)
            findArucoMarkers(frame)
            analyze(corrected_img)
           
           
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            controlQueue.send(ctrl)
        elif key == ord('t'):
            print("Autofocus trigger (and disable continuous)")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            controlQueue.send(ctrl)
        elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
            if key == ord('i'): expTime -= EXP_STEP
            if key == ord('o'): expTime += EXP_STEP
            if key == ord('k'): sensIso -= ISO_STEP
            if key == ord('l'): sensIso += ISO_STEP
            expTime = clamp(expTime, expMin, expMax)
            sensIso = clamp(sensIso, sensMin, sensMax)
            print("Setting manual exposure, time: ", expTime, "iso: ", sensIso)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(expTime, sensIso)
            controlQueue.send(ctrl)
           
           
