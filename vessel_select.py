

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

ims=0
para=True
para1=True
contour=0
ps=0
imggg=0
C=0
parts = []
index = -1
ims1 = 0
ps1=0

def vessel_segment(img, t=8, A=200,L=50):  

    # Green Channel
    g = img[:,:,1]

    #Creating mask for restricting FOV
    _, mask = cv2.threshold(g, 10, 255, cv2.THRESH_BINARY)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.erode(mask, kernel, iterations=3)

    # CLAHE and background estimation
    clahe = cv2.createCLAHE(clipLimit = 3, tileGridSize=(9,9))
    g_cl = clahe.apply(g)
    g_cl1 = cv2.medianBlur(g_cl, 5)
    bg = cv2.GaussianBlur(g_cl1, (55, 55), 0)

    # Background subtraction
    norm = np.float32(bg) - np.float32(g_cl1)
    norm = norm*(norm>0)

    # Thresholding for segmentation
    _, t = cv2.threshold(norm, t, 255, cv2.THRESH_BINARY)

    # Removing noise points by coloring the contours
    t = np.uint8(t)
    th = t.copy()
    contours, hierarchy = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if ( cv2.contourArea(c)< A):
            cv2.drawContours(th, [c], 0, 0, -1)
    th = th*(mask/255)
    th = np.uint8(th)

    # Distance transform for maximum diameter
    vessels = th.copy()
    _,ves = cv2.threshold(vessels, 30, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(vessels, cv2.DIST_L2, 3)
    _,mv,_,mp = cv2.minMaxLoc(dist)

    print("Choose the Blood Vessel Segment and Press Enter to Proceed")

    # Centerline extraction using Zeun-Shang's thinning algorithm
    # Using opencv-contrib-python which provides very fast and efficient thinning algorithm
    # The package can be installed using pip
    thinned = cv2.ximgproc.thinning(th)

    # Filling broken lines via morphological closing using a linear kernel
    kernel = np.ones((1, 10), np.uint8)
    d_im = cv2.dilate(thinned, kernel)
    e_im = cv2.erode(d_im, kernel) 
    num_rows, num_cols = thinned.shape
    for i in range (1, 360//15):
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 15*i, 1)
        img_rotation = cv2.warpAffine(thinned, rotation_matrix, (num_cols, num_rows))
        temp_d_im = cv2.dilate(img_rotation, kernel)
        temp_e_im = cv2.erode(temp_d_im, kernel) 
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -15*i, 1)
        im = cv2.warpAffine(temp_e_im, rotation_matrix, (num_cols, num_rows))
        e_im = np.maximum(im, e_im)

    # Skeletonizing again to remove unwanted noise
    thinned1 = cv2.ximgproc.thinning(e_im)
    thinned1 = thinned1*(mask/255)

    # Removing bifurcation points by using specially designed kernels
    # Can be optimized further! (not the best implementation)
    thinned1 = np.uint8(thinned1)
    thh = thinned1.copy()
    hi = thinned1.copy()
    thi = thinned1.copy()
    hi = cv2.cvtColor(hi, cv2.COLOR_GRAY2BGR)
    thi = cv2.cvtColor(thi, cv2.COLOR_GRAY2BGR)
    thh = thh/255
    kernel1 = np.array([[1,0,1],[0,1,0],[0,1,0]])
    kernel2 = np.array([[0,1,0],[1,1,1],[0,0,0]])
    kernel3 = np.array([[0,1,0],[0,1,1],[1,0,0]])
    kernel4 = np.array([[1,0,1],[0,1,0],[0,0,1]])
    kernel5 = np.array([[1,0,1],[0,1,0],[1,0,1]])
    kernels = [kernel1, kernel2, kernel3, kernel4, kernel5]
    for k in kernels:
        k1 = k
        k2 = cv2.rotate(k1, cv2.ROTATE_90_CLOCKWISE)
        k3 = cv2.rotate(k2, cv2.ROTATE_90_CLOCKWISE)
        k4 = cv2.rotate(k3, cv2.ROTATE_90_CLOCKWISE)
        ks = [k1, k2, k3, k4]
        for kernel in ks:
            th = cv2.filter2D(thh, -1, kernel)
            for i in range(th.shape[0]):
                for j in range(th.shape[1]):
                    if(th[i,j]==4.0):
                        cv2.circle(hi, (j, i), 2, (0, 255, 0), 2)
                        cv2.circle(thi, (j, i), 2, (0, 0, 0), 2)

    thi = cv2.cvtColor(thi, cv2.COLOR_BGR2GRAY)
    cl = thi.copy()
    contours, hierarchy = cv2.findContours(thi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if (c.size<L):
            cv2.drawContours(cl, [c], 0, 0, -1)


    # Centerline superimposed on green channel
    colors = [(100, 0, 150), (102, 0, 255), (0, 128, 255), (255, 255, 0), (10, 200, 10)]
    colbgr = [(193, 182, 255), (255, 0, 102), (255, 128, 0), (0, 255, 255), (10, 200, 10)]

    im = g.copy()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    thc = cl
    thh = thc.copy()
    thh = cv2.cvtColor(thh, cv2.COLOR_GRAY2BGR)
    contours, heirarchy = cv2.findContours(thc, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        
            color = np.random.randint(len(colors))
            cv2.drawContours(im, c, -1, colbgr[color], 2, cv2.LINE_AA)

    d = mv*1.5
    return im, cl, d

# Find the closest contour to the selected point
def cont(ps, j, point, C):

    nonzero = cv2.findNonZero(ps)
    distances = np.sqrt((nonzero[:,:,0] - point[1]) ** 2 + (nonzero[:,:,1] - point[0]) ** 2)
    nearest_index = np.argmin(distances)
    x, y = nonzero[nearest_index][0]
    
    cont,_ = cv2.findContours(ps, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in cont:
        if (cv2.pointPolygonTest(c,(x, y),True)>=0):
            cv2.drawContours(j, c, -1, (255, 255, 255), 4)
            cv2.drawContours(C, c, -1, (255, 255, 255), 1)
            return c

# Draws and returns the selected contour
def draw_c(event,x,y,flags,param):
    global ims, para, contour, ps, imggg, C
    if para==True:
        ims = imggg.copy()
    if event == cv2.EVENT_MOUSEMOVE and para == True:
        cv2.circle(ims,(x,y),4,(255,255,255),2)
        cv2.line(ims,(0, y), (x,y),(0,0,255), 2)
        cv2.line(ims,(x,y),(imggg.shape[1]-1, y),(0,0,255),2)
        cv2.line(ims,(x, 0), (x,y),(0,0,255),2)
        cv2.line(ims,(x,y),(x, imggg.shape[0]-1),(0,0,255),2)
        
    if event == cv2.EVENT_LBUTTONDOWN and para == True:
        para = False
        contour = cont(ps, ims, (y, x), C)

# Selects the minimum distance between a part of contour and the selected point
def find_mindist(ar, point):
    distance = 100000000
    for p in ar:
        d = math.sqrt((p[0][0]-point[0])**2+(p[0][1]-point[1])**2)
        if (d<distance):
            distance = d
    return distance

# selects the contour part closest to the selected point
def select_contour (parts, point):
    idx = -1
    distance = 100000000
    for i,part in enumerate(parts):
        d = find_mindist(part, point)
        if(d<distance):
            distance = d
            idx = i
    return idx

# draws and returns the selected contour part
def draw_part(event,x,y,flags,param):
    global ps1, para1, parts, ims1, index
    if para1==True:
        ims1 = ps1.copy()

    if event == cv2.EVENT_MOUSEMOVE and para1==True:
        cv2.circle(ims1,(x,y),4,(255,255,255),2)
        cv2.line(ims1,(0, y), (x,y),(0,0,255), 2)
        cv2.line(ims1,(x,y),(ps1.shape[1]-1, y),(0,0,255),2)
        cv2.line(ims1,(x, 0), (x,y),(0,0,255),2)
        cv2.line(ims1,(x,y),(x, ps1.shape[0]-1),(0,0,255),2)
        
    if event == cv2.EVENT_LBUTTONDOWN and para1==True:
        para1 = False
        point = (x, y)
        index = select_contour(parts, point)
        for i, part in enumerate(parts):
            if(i!=index):
                 cv2.drawContours(ims1, part, -1, (0, 0, 0), 5)
        cv2.drawContours(ims1, parts[index], -1, (255, 255, 255), 5)

def select(img):
    global ims, para, contour, ps, imggg, C, parts, index, para1, ims1, ps1
    # Segmentation
    im, cl, d = vessel_segment(img)

    # Setting values of the global variables
    ps = cl.copy()
    imggg = im
    j = imggg.copy()
    ims = imggg.copy()
    C = np.zeros(ims.shape, np.uint8)
    contour = None
    para = True
        
    # Selction of contour
    cv2.namedWindow('Blood Vessel Segmented Map', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Blood Vessel Segmented Map', (int(im.shape[1]/2), int(im.shape[0]/2)))
    cv2.moveWindow('Blood Vessel Segmented Map', 40,0) 
    cv2.setMouseCallback('Blood Vessel Segmented Map',draw_c)

    while(1):
        cv2.imshow('Blood Vessel Segmented Map',ims)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('\r'):
            break
    cv2.destroyAllWindows()

    # Finding all the points on the contour 
    Cp = cv2.cvtColor(C, cv2.COLOR_BGR2GRAY)
    point = cv2.findNonZero(Cp)

    # Taking as input the number of parts to which the contour should be divided
    num_parts = input("Choose the No of Segments in which the segment had to be divided(Choose value >=2 and <=5)")
    #print(point.shape)

    print("Choose the Blood Vessel Segment and Press Enter to Proceed")
    parts = np.array_split(point, int(num_parts), axis=0)
    colbgr = [(193, 182, 255), (255, 0, 102), (255, 128, 0), (0, 255, 255), (10, 200, 10)]
    Cparts = np.zeros(C.shape)

    for i, part in enumerate(parts):
        if (i>=5):
            cv2.drawContours(Cparts, part, -1, (255,255,255), 5)
        else:
            cv2.drawContours(Cparts, part, -1, colbgr[i], 5)

    # Global variables
    para1 = True
    ps1 = Cparts
    ims1 = ps1.copy()
    parts = np.array_split(point, int(num_parts), axis=0)


    # At max supports 10 parts
    if (int(num_parts)<=1 or int(num_parts)>10):
        num_parts = 1
        C_parts_selected = parts[0]

        return (C_parts_selected, d)


    # Selecting the part if number of parts <=10
    cv2.namedWindow('User Segment Map', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('User Segment Map', (int(im.shape[1]/2), int(im.shape[0]/2)))
    cv2.moveWindow('User Segment Map', 40,0) 
    cv2.setMouseCallback('User Segment Map', draw_part)

    while(1):
        cv2.imshow('User Segment Map',ims1)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('\r'):
            break
    cv2.destroyAllWindows()



    C_parts_selected = parts[index]


    return C_parts_selected, d,ims1


