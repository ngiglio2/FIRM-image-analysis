# -*- coding: utf-8 -*-
"""
This code trains a model with random-forest classifier.
Needs training images and its masks (annotated images)
"""
#import numpy as np
import cv2
import pandas as pd
#from matplotlib import pyplot as plt
import os
#import dask.dataframe as dd

########################################################
# STEP 1: READ TRAINING IMAGES AND EXTRACT FEATURES
########################################################

# empty dataset for our image features
image_dataset = pd.DataFrame() 
# insert path to the training images
img_path = 'training_images/'


for image in os.listdir(img_path):
    print('reading in ' + str(image))
    df = pd.DataFrame() #temp dataframe for each loop
    
    input_img = cv2.imread(img_path + image)
    
    #if input image RGB convert to grey
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    else:
        raise Exception("the module works only w grayscale and RGB images!")
        
# START ADDING DATA TO DATAFRAME
###############################################################################    
# ADD PIXEL VALUES TO DATAFRAME
    print('adding pixel values of ' + str(image))
    # reshapes it into one single column 
    pixel_values = img.reshape(-1)
    # just labeling columns
    df['Pixel_Value'] = pixel_values 
    df['Image_Name'] = image # capture image name as we read multiple images 
         
###############################################################################   
# GABOR FILTERS 
    print('extracting features of ' + str(image))
   # num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
   # kernels = []
   # for theta in range(4):   #Define number of thetas
   #     theta = theta / 4. * np.pi
   #     for sigma in (1, 3):  #Sigma with 1 and 3
   #         for lamda in np.arange(np.pi / 4, np.pi, np.pi / 4):   #Range of wavelengths
   #            for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
   #             
   #                 gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
   #                print(gabor_label)
   #                 ksize=9
   #                 kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
   #                 kernels.append(kernel)
   #                 #Now filter the image and add values to a new column 
   #                 fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
   #                 filtered_img = fimg.reshape(-1)
   #                 df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
   #                 num += 1  #Increment for gabor column label
                   
    #df = df.drop(columns=['Gabor1' , 'Gabor9' , 'Gabor10' , 'Gabor11' , 'Gabor12' , 'Gabor13', 'Gabor14' , 'Gabor19' , 'Gabor20', 'Gabor25' , 'Gabor33' , 
    #                                     'Gabor34', 'Gabor35', 'Gabor36', 'Gabor37', 'Gabor38' ,'Gabor43', 'Gabor44'])
    #print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
    
                
#######################################################################################################
#OTHER FEATURES     
    import gc #garbage collector to wipe objects after done using
    #CANNY EDGE 
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1) #reshape, creates 1 column
    df['Canny Edge'] = edges1 #Add column to original dataframe
    

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
    
    #GAUSSIAN with sigma=1
    gaussian_img2 = nd.gaussian_filter(img, sigma=1)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    del gaussian_img2, gaussian_img3
    gc.collect()
    
    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    del median_img
    gc.collect()
    
    #update dataframe with features of each image 
    print('done extracting, appending data to dataset')
    image_dataset = image_dataset.append(df)
    del df
    gc.collect()
##########################################################################
# STEP 2: READ LABELED IMAGES (MASKS) AND CREATE ANOTHER DATAFRAME
#         WITH LABEL VALUES AND LABEL FILE NAMES
##########################################################################

mask_dataset = pd.DataFrame() #create empty datafame to capture mask info

# insert path to the image masks 
mask_path = 'training_images_masks/'
#import os
for mask in os.listdir(mask_path): 
    print('reading mask ' + str(mask))
    
    df2 = pd.DataFrame() #temp dataframe to capture info for each respective mask
    input_mask = cv2.imread(mask_path + mask)
    
    
    #if input mask is RGB change to grey
    if input_mask.ndim == 3 and input_mask.shape[-1] == 3:
        label = cv2.cvtColor(input_mask,cv2.COLOR_BGR2GRAY)
    elif input_mask.ndim == 2:
        label = input_mask
    else:
        raise Exception("the module only works with grayscale and RGB images!")
   
    #add pixel values to data frame
    print('adding pixel values of ' + str(mask))
    label_values = label.reshape(-1)
    df2['Label_Value'] = label_values
    df2['Mask_Name'] = mask

    print('appending mask data to dataset')
    mask_dataset = mask_dataset.append(df2) #update mask dataframe with all info from each mask 
    
    del df2 #delete and call gc to remove from RAM
    gc.collect()
##############################################################################
# STEP 3: GET DATA READY FOR RANDOM FOREST CLASSIFIER 
#         COMBINE BOTH DATAFRAMES INTO SINGLE DATASET
##############################################################################

print('combining datasets')
# here we have two datasets, one with all the training images and one with the
# pixel values of the corresponding masks. Now we will just combine them together
dataset = pd.concat([image_dataset, mask_dataset], axis=1) #concatenate both image and mask datasets

# the annotated images either have white(background), gray(defect), or black(not annotated) colors
# since we don't want to confuse the algorithm, we take out all the rows that have 
# pixel value of 0 (black) and keep only the annotated data
dataset_dropped = dataset[dataset.Label_Value != 0]

# traditional machine learning requires X (independent) and Y (dependent) variables. 
# our X is all the training image data, which we will use to predict Y
print('assigning variables X Y')
#ASSIGN TRAINING FEATURES TO X, LABELS TO Y

#X is features for only annotated pixels
X = dataset_dropped.drop(labels = ["Image_Name", "Mask_Name", "Label_Value"], axis=1)
#X_og is features for everything including non-annotated pixels
X_og = dataset.drop(labels = ["Image_Name", "Mask_Name", "Label_Value"], axis=1)
#assign label values to Y (our prediction)
Y = dataset_dropped["Label_Value"].values
del dataset, dataset_dropped
gc.collect()
# here we split the data to train/test. test_size = 0.2 means 20% of our dataset
# will be kept for testing, and 80% of it will be used to train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

#########################################################################
# STEP 4: DEFINE CLASSIFIER AND FIT MODEL WITH OUR TRAINING DATA
#########################################################################
from sklearn.ensemble import RandomForestClassifier
print('executing random forest classifier')
# instantiate model with n number of decision trees
# n_estimators is the number of decision trees, the more you have, the more 
# complex the classification gets and in general the more accurate results. 
# however, it does take up a lot of memory. 10 is on the lower side
model = RandomForestClassifier(n_estimators = 8, random_state = 42)
print('fitting model with training data')
# fit model on training data 
model.fit(X_train, y_train)

del X_train, y_train
gc.collect()
###########################################################################
# STEP 5: ACCURACY CHECK (not necessary)
###########################################################################

from sklearn import metrics
print('predicting model on test data')
prediction_test = model.predict(X_test)
# recall the 20% that we kept for testing; we will now use the algorithm we trained
# and feed it the 20% data to output values. It then compares those values
# to the actual values, which are the pixel values of the annotations. 
# since defects are small in comparison to the background annotations, 
# accuracies are not really useful (usually around 50-70% accuracy)
print("accuracy = ", metrics.accuracy_score(y_test, prediction_test))


#look at which filters are contributing the most to our model to cut down on unnecessary filters
importances = list(model.feature_importances_)
features_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False)
print(feature_imp)

###########################################################################
# STEP 6: SAVE MODEL FOR FUTURE USE
###########################################################################
import joblib
#now that we have trained a model, dump this model (whatever name you want) into a pickle file
print('saving model as cervix fibril detection model')
model_name = 'cervix fibril detection model'
joblib.dump(model,open(model_name, 'wb'))

print('training model complete with 10 estimators, 20% test size')


