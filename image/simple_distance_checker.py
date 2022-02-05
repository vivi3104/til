import sys, os
import cv2
from math import sqrt, fsum, ceil
import numpy as np

def cv_imread(input_file, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    return cv2.imdecode(np.fromfile(input_file, dtype=dtype), flags)

def cv_imwrite(output_image_path, img, params=None):
    result, buffer = cv2.imencode(os.path.splitext(output_image_path)[1], img, params)
    if result:
        with open(output_image_path, mode='wb') as f:
            buffer.tofile(f)
        return True
    else:
        return False

def dist(p0, p1):
    return sqrt( fsum( [(x1 - x0)**2 for x0, x1 in zip(p0, p1)] ) )

def getVector_f(p0, p1, normed=True):
    vec = (p0[0] - p1[0], p0[1] - p1[1])
    distance = dist(p0, p1)
    if normed:
        vec = (float(vec[0])/distance, float(vec[1])/distance)
    return vec, distance

# 2D vector -90 degree rotation
def getPerpendicularVector(vec):
    return (-vec[1], vec[0])

def getCenterPoint(p0, p1):
    center_pt = ((p0[0] + p1[0])//2, (p0[1] + p1[1])//2)
    return center_pt

def process_mouse_event(event, x, y, flags, params):
    input_image   = params['input_image']
    window_name   = params['window_name']
    points        = params['points']
    resize_ratio  = params['resize_ratio']
    output_image_path  = params['output_image_path']
    perpendicular_line_length  = params['perpendicular_line_length']

    if event == cv2.EVENT_LBUTTONDOWN:
        if not (flags & cv2.EVENT_FLAG_CTRLKEY):
            if len(points) > 0:
                if not points[-1] == (x, y):
                    points.append((x, y))
            else:
                points.append((x, y))
                
        
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 0:
            points.pop(-1)
    
    img = input_image.copy()

    # draw points
    if len(points) > 0:
        for i in range(len(points)):
            cv2.circle(img, (points[i][0], points[i][1]), 1, (0, 0, 255), 2)

    # draw pre-line
    if len(points) % 2 == 1:
        last_pt = points[-1]
        if not last_pt == (x, y):
            cv2.line(img, last_pt, (x, y), (0, 255, 0), 1, lineType=cv2.LINE_AA)

            # draw perpendicular line at current point
            center_pt = getCenterPoint(last_pt, (x, y))
            vec_f, distance = getVector_f(last_pt, (x, y), normed=True)
            perpendicular_vector = getPerpendicularVector(vec_f)
            start_pt = (x + int(perpendicular_vector[0] *  perpendicular_line_length//2), y + int(perpendicular_vector[1] *  perpendicular_line_length//2))
            end_pt   = (x + int(perpendicular_vector[0] * -perpendicular_line_length//2), y + int(perpendicular_vector[1] * -perpendicular_line_length//2))
            cv2.line(img, start_pt, end_pt, (0, 255, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(img, "{:.1f} pix".format(distance*resize_ratio), center_pt, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # draw lines for the clicked points
    nLines = len(points)//2
    lines = [ (points[2*j], points[2*j+1]) for j in range(nLines) ]
    for lpt in lines:
        cv2.line(img, lpt[0], lpt[1], (0, 0, 255), 1, lineType=cv2.LINE_AA)
        distance = dist(lpt[0], lpt[1])
        original_distance = distance * resize_ratio
        center_pt = getCenterPoint(lpt[0], lpt[1])
        cv2.putText(img, "{:.1f} pix".format(original_distance), center_pt, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
    
    # save image when CTRL + LEFT CLICK
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            cv_imwrite(output_image_path, img)
            cv2.putText(img, "saving image to: " + output_image_path, (0, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

    # draw cross lines and x,y coordinate in original image scale 
    h, w = img.shape[:2]
    cv2.line(img, (x, 0), (x, h), (255, 0, 0), 1)
    cv2.line(img, (0, y), (w, y), (255, 0, 0), 1)
    original_x = ceil(x * resize_ratio)
    original_y = ceil(y * resize_ratio)
    cv2.putText(img, "({}, {})".format(original_x, original_y), (0, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(window_name, img)

def main():
    '''
    Show clicked 2 points line with distance in pix
    '''
    args = sys.argv
    if len(args) < 2:
        print("Please set input image file path to start")
        return -1
    
    input_image_path = args[1]
    if not os.path.isfile(input_image_path):
        print("Error - Unepxected input path: ", input_image_path)
        return -1

    input_image_path_wo_ext, _ = os.path.splitext(input_image_path)
    output_image_path = input_image_path_wo_ext + '_dist.jpg'
    
    image = cv_imread(input_image_path, cv2.IMREAD_COLOR, np.uint8)
    h = image.shape[0]

    # expect full HD physical display
    h_ratio = 720/float(h)
    scale_down_factor = h_ratio if h_ratio < 1 else 1
    target_image = cv2.resize(image, dsize=None, fx=scale_down_factor, fy=scale_down_factor)

    points = []
    perpendicular_line_length = 30
    window_name = 'mouse_event'
    params = {
        "input_image"  : target_image,
        "window_name"  : window_name,
        "points"       : points,
        "resize_ratio" : 1/scale_down_factor,
        "output_image_path" : output_image_path,
        "perpendicular_line_length" : perpendicular_line_length
    }

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name=window_name, on_mouse=process_mouse_event, param=params)
    cv2.imshow(window_name, target_image)
    cv2.waitKey(0)
    # while True:
    #     key = cv2.waitKey(20)
    #     if (key == 27 or key == ord('q')):
    #         break
    #     elif (key == ord('s')):
    #         # tried to capture image but failed
    #         cv_imwrite(output_image_path, target_image)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
