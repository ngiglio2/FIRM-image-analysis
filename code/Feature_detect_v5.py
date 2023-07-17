# -*- coding: utf-8 -*-
"""
@author: justi

this code uses a trained model to run through images in a folder and detect defects

"""
#%%
#########################################################
# FUNCTION FOR FEATURE EXTRACTION
#########################################################
import numpy as np
import cv2
import pandas as pd
import gc
 
def feature_extraction(img): 
    #create empty data frame
    df = pd.DataFrame()
    #reshape img pixels into one column
    img2 = img.reshape(-1)
    #add column into empty dataframe, label it 'original image'
    df['original image'] = img2

 
    #FIRST SET - GABOR FEATURES 
    #num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    #kernels = []
    #for theta in range(4):   #Define number of thetas
    #    theta = theta / 4. * np.pi
    #    for sigma in (1, 3):  #Sigma with 1 and 3
    #        for lamda in np.arange(np.pi / 4, np.pi, np.pi / 4):   #Range of wavelengths
    #            for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
    #        
    #                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
    #                #                print(gabor_label)
    #                ksize=9
    #                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
    #                kernels.append(kernel)
    #                #Now filter the image and add values to a new column 
    #                fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
    #                filtered_img = fimg.reshape(-1)
    #                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    #print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
     #               num += 1  #Increment for gabor column label
  #  df = df.drop(columns=['Gabor1' , 'Gabor9' , 'Gabor10' , 'Gabor11' , 'Gabor12' , 'Gabor13', 'Gabor14' , 'Gabor19' , 'Gabor20', 'Gabor25' , 'Gabor33' , 
  #                                       'Gabor34', 'Gabor35', 'Gabor36', 'Gabor37', 'Gabor38' ,'Gabor43', 'Gabor44'])
                     
    #del gabor_label, ksize, kernel, kernels, fimg, filtered_img
    gc.collect()
    
    #CANNY EDGE 
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1) #reshape, creates 1 column
    df['Canny Edge'] = edges1 #Add column to original dataframe
    del edges, edges1
    gc.collect()


    from skimage.filters import roberts, sobel, scharr, prewitt

    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    del edge_roberts, edge_roberts1
    gc.collect()

    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    del edge_sobel, edge_sobel1
    gc.collect()


    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    del edge_scharr, edge_scharr1 
    gc.collect()
    
    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    del edge_prewitt, edge_prewitt1
    gc.collect()

    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1
    del gaussian_img, gaussian_img1
    gc.collect()
    
    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    del gaussian_img2, gaussian_img3
    gc.collect()

    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    del median_img, median_img1
    gc.collect()
          
    return df


#%%
###########################################################
# USE TRAINED MODEL TO DETECT DEFECTS 
###########################################################
#import glob
import joblib
from matplotlib import pyplot as plt
import os

#load the model we pickled previously
filename = 'cervix fibril detection model'
print('loading cervix fibril detection model')
load_model = joblib.load(open(filename, 'rb'))

#declare folder where images are (the images i want to analyze)
path = 'testing_images/'

#create a list of all the image file names for the headers of the csv file
propList = []
for file in os.listdir(path):
    propList.append(file)
area_dataset = pd.DataFrame()        
#create empty dataset to append each image's defects' area values 

#iterate through every image in folder 
for file in os.listdir(path): 
    print('reading in file ' + str(file))
    
    input_file = cv2.imread(path + file)
        #if input mask is RGB change to grey
    if input_file.ndim == 3 and input_file.shape[-1] == 3:
        label = cv2.cvtColor(input_file,cv2.COLOR_BGR2GRAY)
    elif input_file.ndim == 2:
        label = input_file
    else:
        raise Exception("the module only works with grayscale and RGB images!")

    img = label
    # extract features from each image
    print('doing extraction on file ' + str(file))
    X = feature_extraction(img)
    #FIT TO MODEL, PREDICT 
    print('loading model and predicting on file ' + str(file))
    result = load_model.predict(X)
    #recall X is dependent variable, 42 columns of feature extracted data
    del X
    gc.collect()

    
    #reshape since result is one column
    segmented2 = result.reshape((img.shape))
    print('saving segmented')
    plt.imsave('segmented/seg.jpg'+ str(file), segmented2, cmap='gray')
    #up to this point it will segment out the defects, but 
    #will be a bunch of dots since it iterates pixel by pixel

    ####################################################################
    #clean up segmented image using morphological operations
    ####################################################################
    print('solidifying image')
    from PIL import Image
    
    MyImg = Image.open('segmented/seg.jpg'+ str(file))
    pixels = MyImg.load()
    
    #since the images are not exactly white, black, or gray, here it changes 
    # those pixels to be strictly white, black or gray
    for i in range(MyImg.size[0]): # for every pixel:
        for j in range(MyImg.size[1]):
            if pixels[i,j] >= (80, 80, 80) and pixels[i,j] <= (160,160,160):
                # change to white if gray defect
                pixels[i,j] = (255, 255 ,255)
            else:
                pixels[i,j] = (0, 0, 0)
    
    open_cv_image = np.array(MyImg)
    open_cv_image = open_cv_image[:,:,0] 
    th = open_cv_image


    #FILL IN HOLES, GET RID OF SOME NOISE 
    blur = cv2.GaussianBlur(th,(7,7),0)
    thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
    
    
    #plt.imsave('dragon3/d3_solid/solid.jpg' + str(file), thresh, cmap='gray')
    #at this point we now have solidifed defects instead of scattered pixels 

    ###########################################################################
    # extract contours and filter out unwanted defects 
    ###########################################################################
    image = thresh
    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # create a binary thresholded image
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw all contours
    image_c = cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
    
    gc.collect()
    
    
    # FILTER BASED ON CIRCULARITY
    # Set up the detector with default parameters.
    
    #detector = cv2.SimpleBlobDetector()
    # Detect blobs.
    #keypoints = detector.detect(image)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    #im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    
    #cv2.imshow("Keypoints", im_with_keypoints)
    #cv2.waitKey(0)



    
    #initialize list for areas of each defect
    area_list = []
    #get image dimensions 
    h, w, _ = image.shape
    size = h*w
    #filter out unwanted areas (too big, too small)
    # here, some detected defects are really tiny specks or really big blobs, 
    # both which are obviously not defects, so an area filter is used to delete those.
    # these values can be played around with, but it basically goes through every
    # defect and says if its area doesn't fall within a certain percentage range of the 
    # entire image area, take it out. 
    #area_min = size * .0002
    #area_max = size * .002
    area_min = 1
    area_max = 10000
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= area_min or area >= area_max:
           cv2.drawContours(image, [cnt], -1, 0, -1, 1)
        else:
            area_list.append(area)
        perimeter = cv2.arcLength(cnt, True)
        
    print(str(file) + ' areas:')
    print(area_list)
    defects_num = len(area_list)
    gc.collect()

    #############################################
    #label each defect using regionprops, output area list 
    from scipy import ndimage
    from skimage import color, measure
    
    if image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    elif input_file.ndim == 2:
        image = image
    else:
        raise Exception("the module only works with grayscale and RGB images!")

    s = [[1,1,1], [1,1,1], [1,1,1]]
    label_mask, num_label = ndimage.label(image, structure = s)
    
    img2 = color.label2rgb(label_mask, bg_label=0)
    # up to here we have a fully segmented image with each defect colored so that
    # we can see what the algorithm thinks is a distinct defect.
    # the reason for this is that some defects may overlap slightly, 
    # a watershed algorithm can be introduced here to separate those instances
    plt.imsave('testing_results/final.jpg' + str(file), img2)
    
    ####################################################
    # use regionprops to extract each area and export to csv file (excel)
    ####################################################
    clusters = measure.regionprops(label_mask, img)
    print(str(file) + ' areas:')
    for prop in clusters:
        print('Label:{} Area: {}'.format(prop.label, prop.area))
        
        
    # pixels to nanometer conversion (for now put as 1, which doesn't change value)
    #####################################
    pixels_to_nm = 1
    ######################################
    
    print('exporting data to csv')
    # create temporary list for areas
    temp_area = []
    for prop in clusters:
        temp_area.append(prop.area*pixels_to_nm**2)
    from pandas import DataFrame

    temp2df = DataFrame(temp_area,columns=[file])
    print (temp2df)
    area_dataset = pd.concat([area_dataset,temp2df], ignore_index=False, axis=1)
    

#export area_dataset to csv(excel) file
area_dataset.to_csv('area_measurements.csv', index = False)
# for easier viewing and future analysis, all the area values will be exported to 
# a csv file (excel) names area_measurements with the image file name as the column header





