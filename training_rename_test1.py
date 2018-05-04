'''
                        
                        This program is the training software for gesture recognition

'''

import pickle, cv2, math, timeit, random, time, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

# image analysis part

def nothing(x):
    pass

def find_contours(filename):
    img_pic = cv2.imread(filename)
    img = cv2.resize (img_pic, (0,0), fx=0.4, fy=0.4) 

    # - SEGMENTATION
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #color convert to gray
    img = cv2.medianBlur(img,5) #Bluring gray image

    # - THRESHOLD
    ret, img_th = cv2.threshold(img_gray,127,255,cv2.THRESH_OTSU) #threshold image
    #img_th2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)#9,6 

    # - CONTOURS
    contour_list, hierarchy = cv2.findContours(img_th,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    return contour_list

def find_features(input,features,filename):
    curv_threshold_value = 0.08
    k = 4
    min_area = 2400
    max_area = 9000
    for count in input:
        arc = cv2.arcLength(count, True)
        vertice = cv2.approxPolyDP(count, 0.01*arc, True)
        vert_area = cv2.contourArea(vertice)
        if vert_area > min_area and vert_area < max_area:
            curvature_chain = []
            cont_ar = np.asarray(count)
            #FEATURES
            ellipse = cv2.fitEllipse(count)
            vertices_feature = len(vertice)
            ellipse_feature = cv2.fitEllipse(count)
            (center,axes,orientation) = ellipse
            max_axis_length_ = max(axes)
            min_axis_length = min(axes)            
            area_feature = cv2.contourArea(count)
            perimeter_feature = cv2.arcLength(count,True)
            area_ratio_feature = perimeter_feature / area_feature
            perimeter_ratio_feature = min_axis_length / perimeter_feature 
            length_feature = len(input)

            for i in range(cont_ar.shape[0]-k):
                num = cont_ar[i][0][1]-cont_ar[i-k][0][1] # y
                den = cont_ar[i][0][0]-cont_ar[i-k][0][0] # x
                angle_prev = -np.arctan2(num,den)/np.pi
            
                num = cont_ar[i+k][0][1]-cont_ar[i][0][1] # y
                den = cont_ar[i+k][0][0]-cont_ar[i][0][0] # x
                angle_next = -np.arctan2(num,den)/np.pi
             
                new_curvature = angle_next-angle_prev
                curvature_chain.append(new_curvature)

            convexity = 0
            concavity = 0
            for i in range(len(curvature_chain)):
                if curvature_chain[i] > curv_threshold_value:
                    convexity += 1
                if curvature_chain[i] < -curv_threshold_value:
                    concavity += 1     

            convexity_ratio_feature = convexity / float(i+1)
            concavity_ratio_feature = concavity / float(i+1)
    feature_values = []

    cnt_er = 0
    for ft in use_features:  #This function checks if features were appended correctly
        # - CHECK FOR MISSING FEATURE, COLLECT & APPEND DATA
        if features_dict[ft] in locals():
            feature=eval(features_dict[ft])
            feature_values.append(feature)     
            print '%s' %(features_dict[ft]), feature
        # - SET FEATURE VARIABLE = FALSE,ERROR & APPEND DATA
        else:
            cnt_er+=1
            feature = False # DIRECT APPROACH
            #feature_values.append(feature)
    
    print '---------------------------------------------------------------'
    
    if cnt_er==len(feature_list):
        feed_r = 0
        print '0000000000000000',cnt_er 
    else:
        feed_r = 1
        print '111111111111111'
    return feature_values,feed_r

# Training part

    #classifier
classifier = KNeighborsClassifier(3)
#file and name folders
path='/home/alf/Desktop/Major1/Code/Test1/final_test1/Gestures/'
gesture_names=['close_hand','open_hand','two']

#features pull
features_dict = ['convexity_ratio_feature','area_ratio_feature','length_feature','perimeter_ratio_feature','vertices_feature']
use_features = [0, 1, 2, 3,4]
feature_list = [features_dict[ft] for ft in use_features]

#  n = examples for each object in each folder
n = 20 

n1 = 0 # Loop Count
sequence = range(n)
random.shuffle(sequence)
feature_space = []
labels = []
model_name= 'test60_new_model'
full_model_name = model_name + '.sav'
model_path = '/home/alf/Desktop/Major1/Code/Test1/Final/' + model_name
my_file = Path(model_name)

print 'Training was began'

for object in range(len(gesture_names)):
    for s in range(n):# Loop n Times (Total = gesture_names * n)#
        files_from_folders = [str(filename)for filename in os.listdir(path+gesture_names[object])]#!!!!!!!!!!!!!!!!!!!! move this line to line 164 should solve the problem
        random_folder = random.randint(1, n)
        n1+=1
        print 'RANDOM',random_folder
        print 'SHAPENAMES',gesture_names[object]
        print ' - Train Count : ',n1
        print ' - Object : ',object
        print ' - FILENAME : ',filename

        filename_upd = gesture_names[object]+'/'+files_from_folders[random_folder]
        #get countours
        current_contour = find_contours(path+filename_upd)
        new_feature_values,feedd = find_features(current_contour,feature_list,filename_upd)  # chain = contours(i)
        if feedd == 1:
            feature_space.append(new_feature_values)
            labels.append(object)

print ' - Labels : ',labels
print ' - Space : ',feature_space

# - TRAIN MODEL
classifier.fit(np.asarray(feature_space), np.asarray(labels))# - X_train

    
# - SAVE MODEL TO FILE
pickle.dump(classifier, open(model_name, 'wb'))
if my_file.is_file():
    print 'MODEL SAVED'
else:
    print 'MODEL NOT SAVED'