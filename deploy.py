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

# - CONTOURS
def get_contours(folder,name,col_ind):
    filename = folder + name# + '.png'
    print 'FALSE'
    img = cv2.imread(filename)
    print filename
    # - SEGMENTATION
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,5)
    # - THRESHOLD
    ret, img_th = cv2.threshold(img_gray,127,255,cv2.THRESH_OTSU)
    img_th2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # - CONTOURS
    contour_list, hierarchy = cv2.findContours(img_th,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contour=[contour_list,hierarchy]

    return contour#,img
    #else:
    #    get_contours(filn,folder,name,col_ind)

def find_features(input_mod,features,filname):
    input_contour=input_mod[0]
    h=input_mod[1]
    curvature_threshold = 0.08
    polygon_tolerance = 0.04
    k = 4

    for cnt in input_mod:#input_mod - camera, input_contour - train !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
        # - FIND VERTICES
        arc = cv2.arcLength(cnt, True)
        vert = cv2.approxPolyDP(cnt, 0.01*arc, True)  
        vrt_area = cv2.contourArea(vert)
        # - LIMIT SIZE
        if vrt_area > 700 and vrt_area < 155000:
            #print 'AREA FOUND'
            curvature_chain = []
            cont_ar = np.asarray(cnt)
            
            if cv2.isContourConvex(vert):
                vertices = len(vert)
                #print 'vertices : ',vertices
            
            # - AXIS FEATURES
            ellipse = cv2.fitEllipse(cnt)
            (center,axes,orientation) = ellipse
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            axes_ratio = minoraxis_length/majoraxis_length            
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
            area_ratio = perimeter / area
            perimeter_ratio = minoraxis_length / perimeter 
            epsilon = polygon_tolerance*perimeter
            vertex_approx = 1.0 / len(cv2.approxPolyDP(cnt,epsilon,True))
            length = len(input_mod)
            
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
            #print 'concavity_ratio : ',concavity_ratio
            #print 'convexity_ratio : ',convexity_ratio
            
            # - DRAW CONTOURS (TRAINING)
            #img = cv2.imread('/home/afonso/Desktop/Major/Code/Test1/Final/images4/'+filname)
            #print 'brrrrrrrrrrr3'
            #cv2.drawContours(filname,[cnt],0,(0,255,0),4)# Draw from Cont --------- cv2.drawContours(img
            #cv2.imshow('aaa',filname) -------------------------------cv2.imshow('aaa'
            #cv2.waitKey()
            

            #feature_values = [eval(ft) for ft in features]
        #else:
            # - NO SUITABLE AREA FOUND
            #print 'NO SUITABLE AREA FOUND'
            
    feature_values=[]

    for ft in use_features:
        # - CHECK FOR MISSING FEATURE, COLLECT & APPEND DATA
        if features_dict[ft] in locals():
            feature=eval(features_dict[ft])
            feature_values.append(feature)
            
        # - SET FEATURE VARIABLE=FALSE,ERROR & APPEND DATA
        else:
            feature = False # DIRECT APPROACH
            #exec(features_dict[ft]+'=False')
            #feature=eval(features_dict[ft])
            feature_values.append(feature)
            
            # - MISSING FEATURE
            #if feature==False:
                #print 'Error : '+str(features_dict[ft])+' - '+str(ft)

    
    return feature_values

# - Start
#start_time = timeit.default_timer()

# - CLASSIFIER
classifier = KNeighborsClassifier(3)
classifier_nm = classifier

# - INIT
pth='/home/alf/Desktop/Major1/Code/Test1/Final/images5'
logfile = open('log.txt', 'w')
#shape_names=[c for c in os.listdir(pth) if os.path.isdir(pth+c)]
shape_names=['close','open','time']
# - Features
features_dict = ['axes_ratio','concavity_ratio','convexity_ratio','area_ratio','vertex_approx','length','perimeter_ratio','vertices']
use_features = [0, 1, 2, 3, 4, 5, 6, 7] # 0..6
feature_list = [features_dict[ft] for ft in use_features]

#Load model
model_name = '/home/alf/Desktop/Major1/Code/Test1/Final/test_new_model.sav'
my_file = Path(model_name)
classifier = pickle.load(open(model_name, 'rb'))
# - GUI SETUP
cv2.namedWindow('Input')
cv2.createTrackbar('A','Input',127,255,nothing)
cv2.createTrackbar('A1','Input',127,255,nothing)
cv2.createTrackbar('B','Input',11,20,nothing)
cv2.createTrackbar('C','Input',2,20,nothing)

# CAMERA INPUT INIT
vc = cv2.VideoCapture(1)
print(vc.isOpened())
time.sleep(2) # warm-up camera
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    print 'hey'
    time.sleep(5)
else:
    rval = False

# CAMERA INPUT LOOP START
while rval:
    rval, frame = vc.read()

    # - GUI UPDATE
    a = cv2.getTrackbarPos('A','Input')
    a1 = cv2.getTrackbarPos('A1','Input')
    b = cv2.getTrackbarPos('B','Input')
    c = cv2.getTrackbarPos('C','Input')
    #b+=1 if b % 2 == 0 or b == 0 else b==b
    if b % 2 == 0 or b == 0: #even
        b+=1
    
    # - IMAGE PROCESSING
    img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)# - GRAY
    img = cv2.medianBlur(img_gray,5)# - BLUR
    #print img.shape
    
    # - THRESHOLD
    ret, img_th = cv2.threshold(img_gray,a,a1,cv2.THRESH_OTSU)
    #img_th2 = cv2.adaptiveThreshold(img,a,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,b,c)
    # - CONTOURS
    ##contour_list, hierarchy = cv2.findContours(img_th2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    ##n_obj = len(contour_list)# - NUMBER OF CONTOURS IN CURRENT LOOP
    #print ' - n_obj : ', n_obj

    # - FEATURES EXTRACTION
    ##new_feature_values = find_features(contour_list,feature_list,frame)  # chain = contours(i)
    ##temp = np.asarray(new_feature_values).reshape(1, -1)
   
    # - OUTPUT
    ##prob = classifier.predict_proba(temp)
    ##prob2 = classifier.predict(temp)
    ##classif_rate = np.mean(prob2.ravel()) * 100
    ##pr=prob[0]
    ##print 'result1 : '+'{:.2%}'.format(float(pr[prob2]))
    ##print 'Guess: ', shape_names[np.argmax(prob)]
    ##print 'prob',prob
    ##print 'prob2',prob2
    ##print 'predict : ',classif_rate
    ##print ["%0.6f" % p for p in prob[0]], "%0.2f" % np.mean(prob[0])
    #9,4
    cv2.imshow("Input", frame)
    cv2.imshow("normal_thresh", img_th)
    ##cv2.imshow("adaptive_thresh", img_th2)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyAllWindows("preview")
cv2.release.vc

