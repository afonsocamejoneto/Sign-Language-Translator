import pickle, cv2, math, timeit, random, time, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

# - CONTOURS
def get_contours(filename):
    img_pic = cv2.imread(filename)
    img = cv2.resize (img_pic, (0,0), fx=0.3, fy=0.3) 
    cv2.imshow('resize' ,img)
    # - SEGMENTATION
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,5)

    # - THRESHOLD
    ret, img_th = cv2.threshold(img_gray,127,255,cv2.THRESH_OTSU)
    img_th2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,6)#11,2

    # - CONTOURS
    contour_list, hierarchy = cv2.findContours(img_th,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    return contour_list

def find_features(input_mod,features,filname):
    curvature_threshold = 0.08
    polygon_tolerance = 0.04
    k = 4
    for cnt in input_mod:
        # - FIND VERTICES
        arc = cv2.arcLength(cnt, True)
        vert = cv2.approxPolyDP(cnt, 0.01*arc, True)  
        vrt_area = cv2.contourArea(vert)
        # - LIMIT SIZE
        if vrt_area > 2000 and vrt_area < 9000:
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

            # - DRAW CONTOURS (TRAINING)
            img = cv2.imread('/home/alf/Desktop/Major1/Code/Test1/final_test1/Gestures/'+filname)
            img = cv2.resize (img, (0,0), fx=0.3, fy=0.3)
            cv2.drawContours(img,[cnt],-1,(0,255,0),4)# Draw from Cont --------- cv2.drawContours(img
            print filname 
            cv2.imshow('Contours',img) q
            cv2.waitKey()
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
            feature_values.append(feature)

    return feature_values

# - Start --------------------------------------------------------------------------------------------------------------------------------

# - CLASSIFIER
classifier = KNeighborsClassifier(3)
# - INIT
pth='/home/alf/Desktop/Major1/Code/Test1/final_test1/Gestures/'
shape_names=['close_hand','open_hand','two']
#shape_names=['letter_p','letter_c','letter_g']
# - Features
features_dict = ['axes_ratio','concavity_ratio','convexity_ratio','area_ratio','vertex_approx','length','perimeter_ratio','vertices']
use_features = [0, 1, 2, 3, 4, 5, 6, 7] # 0..6
feature_list = [features_dict[ft] for ft in use_features]

# - World - 10 examples for each object
n = 11


n1 = 0 # Loop Count
sequence = range(n)
random.shuffle(sequence)
feature_space = []
labels = []
model_name = '/home/alf/Desktop/Major1/Code/Test1/Final/test_new_model.sav'
my_file = Path(model_name)

# - CHECK FOR & TRAIN MODEL ------------------
print 'brrrrrrrrr2'
# - TRAIN LOOP
for obj in range(len(shape_names)):# Loop 6 Times - Letters for filenames######
    files_frm_fldrs = [str(filename)for filename in os.listdir(pth+shape_names[obj])]#!!!!!!!!!!!!!!!!!!!!!!!!!!
    for s in range(n):# Loop 10 Times (Total 60)
        rnd_f = random.randint(1, n)
        n1+=1
        print 'RANDOM',rnd_f
        print 'SHAPENAMES',shape_names[obj]
        print 'filesfromfold',files_frm_fldrs[rnd_f]

        print ' - Train Count : ',n1
        print ' - Object : ',obj
        print ' - FILENAME : ',filename

        filename_upd = shape_names[obj]+'/'+files_frm_fldrs[rnd_f]
        current_contour = get_contours(pth+filename_upd)# Get Countours
        new_feature_values = find_features(current_contour,feature_list,filename_upd)  # chain = contours(i)
        feature_space.append(new_feature_values)
        labels.append(obj)

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