import sys
import cv2
from math import dist

def getVector_f(p0, p1, normed=True):
    vec = [p0[0] - p1[0], p0[1] - p1[1]]
    __distance = dist(p0, p1)
    if normed:
        vec = [float(vec[0])/__distance, float(vec[1])/__distance]
    return vec, __distance

# 2D vector -90 degree rotation
def getPerpendicularVector(vec):
    return [-vec[1], vec[0]]

def getCenterPoint(p0, p1):
    center_pt = [(p0[0] + p1[0])//2, (p0[1] + p1[1])//2]
    return center_pt

def process_mouse_event(event, x, y, flags, params):
    raw_image     = params['image']
    window_name = params['window_name']
    points      = params['points']
    num_points  = params['num_points']
    perpendicular_line_length  = params['perpendicular_line_length']

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < num_points:
            points.append([x, y])
        
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 0:
            points.pop(-1)
    
    image = raw_image.copy()
    h, w = image.shape[:2]
    cv2.line(image, (x, 0), (x, h), (255, 0, 0), 1)
    cv2.line(image, (0, y), (w, y), (255, 0, 0), 1)

    # draw points
    if len(points) > 0:
        for i in range(len(points)):
            cv2.circle(image, (points[i][0], points[i][1]), 1, (0, 0, 255), 2)

    # draw pre-line
    if len(points) % 2 == 1:
        last_pt = points[-1]
        if not last_pt == [x, y]:
            cv2.line(image, last_pt, [x, y], (0, 255, 0), 1, lineType=cv2.LINE_AA)

            # draw perpendicular line at current point
            center_pt = getCenterPoint(last_pt, [x,y])
            vec_f, distance = getVector_f(last_pt, [x, y], normed=True)
            perpendicular_vector = getPerpendicularVector(vec_f)
            start_pt = [x + int(perpendicular_vector[0] *  perpendicular_line_length//2), y + int(perpendicular_vector[1] *  perpendicular_line_length//2)] 
            end_pt   = [x + int(perpendicular_vector[0] * -perpendicular_line_length//2), y + int(perpendicular_vector[1] * -perpendicular_line_length//2)]
            cv2.line(image, start_pt, end_pt, (0, 255, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(image, "{:.1f} pix".format(distance), center_pt, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # draw lines for the clicked points
    nLines = len(points)//2
    lines = [ [points[2*j], points[2*j+1]] for j in range(nLines) ]
    for lpt in lines:
        cv2.line(image, lpt[0], lpt[1], (0, 0, 255), 1, lineType=cv2.LINE_AA)
        distance = dist(*lpt)
        center_pt = getCenterPoint(lpt[0], lpt[1])
        cv2.putText(image, "{:.1f} pix".format(distance), center_pt, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(image, "({0}, {1})".format(x, y), (0, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(window_name, image)

def main():
    '''
    measure distance between clicked 2 points
    '''
    args = sys.argv
    if len(args) < 2:
        print("Please set input image file path to start")
        return -1
    
    input_image_path = args[1]
    
    image = cv2.imread(input_image_path)
    h = image.shape[0]

    # expect full HD physical display
    h_ratio = 720/h
    scale = h_ratio if h_ratio < 1 else 1
    target_image = cv2.resize(image, dsize=None, fx=scale, fy=scale)

    points = []
    num_points = 100
    perpendicular_line_length = 30
    window_name = 'mouse_event'
    params = {
        "image" : target_image,
        "window_name" : window_name,
        "points" : points,
        "num_points" : num_points,
        "perpendicular_line_length" : perpendicular_line_length
    }

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name=window_name, on_mouse=process_mouse_event, param=params)
    cv2.imshow(window_name, target_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
