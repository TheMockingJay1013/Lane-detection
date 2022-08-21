import cv2
import numpy as np
SLOPE1=SLOPE2=Y_INT1=Y_INT2=0


#the region function is to limit the area of focus in the frame to the portion we want it to 
#(basically it will focus only the portion where the lane could be )

def region(image):
    height, width = image.shape
    triangle = np.array([
                       [(20, height), (450, 315),(500,315), (width, height)]
                       ])
    
    mask = np.zeros_like(image)
    
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask




#this function is used to obtain 2 points(end points of a lane) from a known line
def make_points(image, average,side,SLOPE,Y_INT):
    if hasattr(average, '__iter__'):
        slope, y_int = average
    else :
        if side=="left":
            slope=SLOPE
            y_int=Y_INT
        elif side=="right":
            slope=SLOPE
            y_int=Y_INT
    print(slope , y_int)
    y1 = image.shape[0]
    y2 = int(y1 * (2/3))
    
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return x1, y1, x2, y2,slope,y_int




#reading the video
vid = cv2.VideoCapture('Level 1.mp4')
 



while(True):
    ret, frame = vid.read() #obtaining each frame as an image
    copy = np.copy(frame)
    
    #this part converts the video frame to greyscale , checks for white and does the canny edge detection on it
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    mask = cv2.inRange(grey,200,255)
    edges = cv2.Canny(mask, 190, 200)
    edges=region(edges)  # the region function limits the area of focus to the portion where the required lane is


    # this part converts the video frame to HSV form and checks for yellow and does canny edge detection on it
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_yellow=np.array([20, 100, 100], dtype = "uint8")
    upper_yellow=np.array([30, 255, 255], dtype = "uint8")
    yellow_mask=cv2.inRange(hsv,lower_yellow,upper_yellow)
    yellow_edge=cv2.Canny(yellow_mask,150,200)
    yellow_edge=region(yellow_edge)  # the region function limits the area of focus to the portion where the required lane is



    # this function is for taking a union of the data from the greyscale and hsv frames (white and yellow lanes)
    edges=cv2.bitwise_or(edges,yellow_edge)
    
    
     

    lines = cv2.HoughLinesP(edges, 2, np.pi / 180 , threshold=50 ,maxLineGap=150 , minLineLength=10 )
    left = []
    right = []
    if lines is not None:
        for line in lines:

            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            
            slope = parameters[0]
            y_int = parameters[1]
            

            
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))
                
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    #create lines based on averages calculates
    X1,Y1,X2,Y2,SLOPE1,Y_INT1 = make_points(frame, left_avg,"left",SLOPE1,Y_INT1)
    X3,Y3,X4,Y4,SLOPE2,Y_INT2 = make_points(frame, right_avg, "right",SLOPE2,Y_INT2)
                    
    pts=np.array([[X1,Y1],[X3,Y3],[X4,Y4],[X2,Y2]])
    cv2.drawContours(frame, [pts], -1, color=(0, 255, 0), thickness=cv2.FILLED)
    image=cv2.addWeighted(copy, 1, frame, 0.5, 0.0)


    #displaying all the forms of video (only for analysing purpose)
    # cv2.imshow('edges', edges)
    # cv2.imshow('mask',mask)
    # cv2.imshow('grey',grey)
    # cv2.imshow('hsv',hsv)
    cv2.imshow('frame',image)

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break


#closes all the windows when the program ends
vid.release()
cv2.destroyAllWindows()

