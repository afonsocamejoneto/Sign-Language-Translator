import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
import cPickle as pickle
import time

# - GUI TRACKBAR PASS FUNCTION
def nothing(x):
    pass

def feature_extraction(input_mod,features,filname):
    input_contour=input_mod[0]
    h=input_mod[1]
    curvature_threshold = 0.08 # constant values for features calculations
    polygon_tolerance = 0.05 # constant values for features calculations
    k = 4

    for contour in input_mod:#input_mod - camera, input_contour - train !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # - FIND VERTICES
        arc = cv2.arcLength(contour, True)
        contour_vertices = cv2.approxPolyDP(contour, 0.01*arc, True)
        vertices__contour_area = cv2.contourArea(contour_vertices)
        # - LIMIT SIZE
        if vertices__contour_area > 18000 and vertices__contour_area < 55000: # the area_feature changes depding on the size of images
            curvature_chain = []
            cont_ar = np.asarray(contour)
            vertices = len(contour_vertices)

           ##FEATURE_extraction_algorithms
            ellipse = cv2.fitEllipse(contour)
            (center,axes,orientation) = ellipse

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




            # - DRAW CONTOURS (TRAINING)
            #rval, img2 = vc.read()
            #img_gary_draw = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            cv2.drawContours(filname,[contour],-1,(0,255,0),4)# Draw from Cont --------- cv2.drawContours(img
            #cv2.imshow('aaa',img_gary_draw) #-------------------------------cv2.imshow('aaa'


    feature_values=[]


    # - CHECK FOR MISSING FEATURE, CO
    counting_error = 0
    for ft in nr_features:

        # - CHECK FOR MISSING FEATURE, COLLECT & APPEND DATA
        if features_list[ft] in locals():
            feature=eval(features_list[ft])
            feature_values.append(feature)
            #print '%s' %(features_list[ft]), feature
        # - SET FEATURE VARIABLE=FALSE,ERROR & APPEND DATA
        else:
            counting_error+=1
            feature = False # DIRECT APPROACH
            feature_values.append(feature)
    if counting_error== len(features_list_array):
        feed_r = 0
    else:
        feed_r = 1
    return feature_values,feed_r

def most_common (lst):
    return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[1])[0]

# - Start -------------------------------------------------------------------------------------------------------------------------------------


#start_time = timeit.default_timer()

# - CLASSIFIER
classifier = KNeighborsClassifier(3)
classifier_nm = classifier

# - INIT
pth='/home/alf/Desktop/Major1/Code/Test1/Final/Gestures'
shape_names=['cls','opn','two']

# - Features
features_list = ['axes_ratio_feature','concavity_ratio','convexity_ratio','area_ratio_feature','vertex_approx_feature','length_feature','perimeter_ratio_feature','vertices']
nr_features = [0, 1, 2, 3, 4, 5, 6, 7] # 0..6
features_list_array = [features_list[ft] for ft in nr_features]

#Load model
model_name = '/home/alf/Desktop/Major1/Code/Test1/Final/test_new_model07.sav'
my_file = Path(model_name)
classifier = pickle.load(open(model_name, 'rb'))

# CAMERA INPUT INIT
vc = cv2.VideoCapture(1)
print(vc.isOpened())
time.sleep(1) # warm-up camera

if vc.isOpened(): # tries to get the first frame
    rval, frame = vc.read()
    print 'Capture first frame'
    time.sleep(1)
else:
    rval = False

counter=1
prob_temp=[]
# CAMERA INPUT LOOP START
while rval:
    rval, frame = vc.read()

    # - IMAGE PROCESSING
    picture_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)# - GRAY

    # - THRESHOLD
    ret, picture_thresh = cv2.threshold(picture_gray,127,255,cv2.THRESH_OTSU)
    #img_th2 = cv2.adaptiveThreshold(img,a,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,b,c)

    # - CONTOURS
    contour_list, hierarchy = cv2.findContours(picture_thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    #Debuging
    ##for x in contour_list:
        ##cv2.drawContours(frame,[x],-1,(0,255,0),4)
        #cv2.drawContours(img_resize,[contour],-1,(0,255,0),4)

    n_obj = len(contour_list)# - NUMBER OF CONTOURS IN CURRENT LOOP
    #print ' - n_obj : ', n_obj

    # - FEATURES EXTRACTION
    new_feature_values,feed_rr = feature_extraction(contour_list,features_list_array,frame)  # chain = contours(i)
    if feed_rr == 1:
        temp = np.asarray(new_feature_values).reshape(1, -1)
        counter+=1

        if counter == 11:
            counter = 0

        # - OUTPUT
        prob_percent = classifier.predict_proba(temp)
        prob_name = classifier.predict(temp)
        classif_rate = np.mean(prob_percent.ravel()) * 100
        prob=prob_percent[0]

        #print '---------------------------------------------------------------'
        ##print 'result1 : '+'{:.2%}'.format(float(prob[prob_name]))
        ##print 'Guess: ', shape_names[np.argmax(prob_percent)]
        ##print 'prob_percent',prob_percent
        ##print 'predict : ',classif_rate
        #print ["%0.6f" % p for p in prob_percent[0]], "%0.2f" % np.mean(prob_percent[0]

        #print '--------------------Common result----------------------------------------'
        common = []
        for i in range(counter):
            common.append(shape_names[np.argmax(prob_percent)])
        ##print 'common', common
        ##print'lengtnh', len(common)
        ##print 'ranger',range(counter)

        if (len(common)+1) == (counter - 1):
            #print 'common', common
            #print(np.bincount(x).argmax())
            print 'The most occurated guess is :', most_common(common)



    cv2.imshow("Input", frame)

    #cv2.imshow("normal_thresh", picture_thresh)
    ##cv2.imshow("adaptive_thresh", img_th2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
vc.release()
