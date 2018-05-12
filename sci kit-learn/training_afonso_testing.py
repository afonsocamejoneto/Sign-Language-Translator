import pickle, cv2, math, timeit, random, time, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

def nothing(x):
    pass

# - Find  CONTOURS function
def find_contours(filename):

    picture = cv2.imread(filename) # picture to read

    # - color converstion
    picture_gray = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)

    # - THRESHOLD
    ret, picture_thresh = cv2.threshold(picture_gray,127,255,cv2.THRESH_OTSU)
    ##picture_thresh2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)#9,6

    # - find CONTOURS
    contour_list, hierarchy = cv2.findContours(picture_thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    return contour_list

def feature_extraction(input_mod,features,filename):
    curvature_threshold = 0.08 # constant values for features calculations
    polygon_tolerance = 0.05 # constant values for features calculations
    k = 4

    for contour in input_mod:
        # - FIND VERTICES
        arc = cv2.arcLength(contour, True)
        contour_vertices = cv2.approxPolyDP(contour, 0.01*arc, True)
        vertices__contour_area = cv2.contourArea(contour_vertices)
        # - LIMIT SIZE
        if vertices__contour_area > 18000 and vertices__contour_area < 55000:
            curvature_chain = []
            cont_ar = np.asarray(contour)
            vertices = len(contour_vertices)

         ##FEATURE_extraction_algorithms

            ellipse_feature = cv2.fitEllipse(contour)
            (center,axes,orientation) = ellipse_feature

            majoraxis_length_feature = max(axes)
            minoraxis_length_feature = min(axes)
            axes_ratio_feature = minoraxis_length_feature/majoraxis_length_feature



            area_feature = cv2.contourArea(contour)
            perimeter_feature = cv2.arcLength(contour,True)

            area_ratio_feature = perimeter_feature / area_feature
            perimeter_ratio_feature = minoraxis_length_feature / perimeter_feature

            epsilon_feature = polygon_tolerance*perimeter_feature
            vertex_approx_feature = 1.0 / len(cv2.approxPolyDP(contour,epsilon_feature,True))
            length_feature = len(input_mod)

            ##### begin of Eris Chintelli code
            # CURVATURE & CONVEXITY
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
                if curvature_chain[i] > curvature_threshold:
                    convexity += 1
                if curvature_chain[i] < -curvature_threshold:
                    concavity += 1

            convexity_ratio = convexity / float(i+1)
            concavity_ratio = concavity / float(i+1)

            ##### end of Eris Chinchilli code

            '''
            crn_f = cv2.imread('/home/alf/Desktop/Major1/Code/Test1/Final/Gestures/'+filename)
            crn_f = cv2.cvtColor(crn_f, cv2.COLOR_BGR2GRAY)
            crn = np.float32(crn_f)
            crn = cv2.cornerHarris(crn,2,3,0.04)

            #hull = cv2.convexHull(contour)
            #hull_area = cv2.contourArea(hull)
            #solidity = area_feature / float(hull_area)
            '''
            # - DRAW CONTOURS (TRAINING)
            ##img_draw = cv2.imread('/home/alf/Desktop/Major1/Code/Test1/Final/Gestures/'+filename)
            #img_resize = cv2.resize(img_draw,(0,0),fx=0.3 , fy=0.3)
            ##cv2.drawContours(img_draw,[contour],-1,(0,255,0),4)
            ##cv2.imshow('contours',img_draw)
            #cv2.waitKey()

    feature_values=[]

    # - CHECK FOR MISSING FEATURE, CO
    counting_error = 0#COUNTING ERROR
    for ft in nr_features:

        # - CHECK FOR MISSING FEATURE, COLLECT & APPEND DATA
        if features_list[ft] in locals():
            feature=eval(features_list[ft])
            feature_values.append(feature)
            print '%s' %(features_list[ft]), feature

        # - SET FEATURE VARIABLE=FALSE,ERROR & APPEND DATA
        else:
            counting_error+=1
            feature = False # DIRECT APPROACH
            feature_values.append(feature)
    if counting_error==len(features_list_array):
        feed_r = 0
    else:
        feed_r = 1
    return feature_values,feed_r

# - Start --------------------------------------------------------------------------------------------------------------------------------

# - CLASSIFIER
classifier = KNeighborsClassifier(3)

# - INIT
pth='/home/alf/Desktop/Major1/Code/Test1/Final/Gestures/'
shape_names=['cls','opn','two']

# - Features
features_list = ['axes_ratio_feature','concavity_ratio','convexity_ratio','area_ratio_feature','vertex_approx_feature','length_feature','perimeter_ratio_feature','vertices']
nr_features = [0, 1, 2, 3, 4, 5, 6, 7] # ideally shound be nr_features= len(features_list)
features_list_array = [features_list[ft] for ft in nr_features]

# - World - nr_pic_folder examples for each object
#nr_pic_folder = 115 # number of training picture per folder
n1 = 0 # Loop Count
feature_space_values = []
labels = []
model_name = '/home/alf/Desktop/Major1/Code/Test1/Final/test_new_model08.sav'
my_file = Path(model_name)
tots_loop = 0
last_cnt = 0

# - CHECK FOR & TRAIN MODEL
print 'Training was began'
# - TRAIN LOOP
for folder in range(len(shape_names)):# Loop shape_names Times ######

    nr_pic_folder=115
    last_cnt = 0
    while last_cnt < nr_pic_folder:
    #for s in range(nr_pic_folder):# Loop nr_pic_folder Times (Total = shape_names x nr_pic_folder) # Do not use
        tots_loop+=1
        files_frm_fldrs = [str(filename)for filename in os.listdir(pth+shape_names[folder])]#
        random_folder = random.randint(1, nr_pic_folder)
        filename_update = shape_names[folder]+'/'+files_frm_fldrs[random_folder]

        #print ' - random_folder :',random_folder
        print ' - SHAPE NAME :',shape_names[folder]
        print ' - Train Count : ',n1
        print ' - last_cnt : ',last_cnt
        print ' - tots : ',tots_loop
        print ' - Folder : ',folder
        print ' - filename_update :',filename_update

        current_contour = find_contours(pth+filename_update)
        train_feature_values,feedd = feature_extraction(current_contour,features_list_array,filename_update)  # chain = contours(i)
        if feedd == 1:
            n1+=1
            last_cnt+=1
            feature_space_values.append(train_feature_values)
            labels.append(folder)
        else:
            nr_pic_folder+=1
print ' - Labels : ',labels
print ' - Space : ',feature_space_values

# - TRAIN MODEL
classifier.fit(np.asarray(feature_space_values), np.asarray(labels))# - X_train

# - SAVE MODEL TO FILE
pickle.dump(classifier, open(model_name, 'wb'))

if my_file.is_file():
    print 'MODEL SAVED : NO erros occur'
else:
    print 'MODEL NOT SAVED : Something happen'
