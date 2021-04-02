import cv2
import numpy as np

'''
    input: 
        + image: we need to align
        + expand_pixel: is the pixel we need when we align item and expand
        (x0,y0) ------------------- (x1, y1)
            |                           |
            |                           |
            |                           |
        (x2,y2) ------------------- (x3, y3)
        + corner: list of corner like on this rectangle
    output: image array is aligned
'''


def align_item(image, expand_pixel, corner):
 
    rows,cols,ch = image.shape
    pts1 = np.float32([corner[0], corner[1], corner[2], corner[3]])
    width_transform = np.abs(corner[0][0] - corner[1][0]) 
    height_transform = np.abs(corner[0][1] - corner[2][1])
    print(width_transform, height_transform)
    pts2 = np.float32([[0,0],[width_transform,0],[0,height_transform],[width_transform,height_transform]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(image,M,(width_transform, height_transform))
    return dst
    # cv2.imwrite('../data/output2.jpg', dst)

if __name__=='__main__':
    img = cv2.imread('../data/cool.png')
    corner = [[5,6], [51,9], [3,37], [48,40]]
    align_item(img, 5, corner)