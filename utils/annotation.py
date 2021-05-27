import cv2
import numpy as np

def draw_points(im_array, coords):
    '''
    Draws the specific point on the image
    Used mostly to track a specific keypoint
    Returns the annotated image
    '''
    im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    for x, y in coords:
        im_array = cv2.circle(im_array, (x, y), 5, [0, 0, 255], -1)
        
    return im_array

def get_part_coords(json, value):
    '''
    Gets the coordinates of the specific point to track. Use pifpaf_keypoints for reference on the 
    key of the body part to be tracked
    '''
    kp = json['keypoints']
    x, y = kp[value*3], kp[value*3+1]
    coords = [(int(x), int(y)),]
    return coords

def draw_bbox(im_array, coords, score, category_id):
    '''
    Draws a bbox on the image
    '''
    (x1, y1), (x2, y2) = coords
    im_array = cv2.rectangle(im_array, (x1, y1), (x2, y2), (255, 0, 255), 1)
    im_array = cv2.putText(im_array, "{}: {}".format(category_id, score), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0 , 255))
    return im_array

def annotate_bbox(im_array, json, min_score: int=0.25, desired_classes:list = [1,]):
    '''
    Checks json to make sure it should be annotated
    Bbox format: 
    - x (top left)
    - y (top left)
    - width
    - height
    '''
    score = json['score']
    category_id = json['category_id']

    if (score < min_score) or (category_id not in desired_classes):
        return im_array
    
    # ELSE draw the box
    bbox = json['bbox']
    print(bbox)
    x1, y1, w, h = list(map(lambda x: int(x), bbox))
    x2 = x1 + w
    y2 = y1 + h

    return draw_bbox(im_array, ((x1, y1), (x2, y2)), score, category_id)