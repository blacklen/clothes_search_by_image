from PIL import Image
import cv2
import glob
import os
import numpy as np


def resize_image(img_path, img_width, img_height):
    img = cv2.imread(img_path)
    scale_img = np.full((img_height, img_width, 3), (255, 255, 255))
    h, w, _ = img.shape

    if h == img_height and w == img_width:
        return img
    scale_h = img_height / h
    scale_w = img_width / w
    if scale_h > scale_w:
        scale = scale_w
    else:
        scale = scale_h

    width = int(w * scale)
    height = int(h * scale)

    x_offset = int((scale_img.shape[0] - height) / 2 - 1)
    y_offset = int((scale_img.shape[1] - width) / 2 - 1)

    if x_offset < 0:
        x_offset = 0
    if y_offset < 0:
        y_offset = 0

    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    scale_img[x_offset:x_offset + height, y_offset:y_offset + width] = resized

    return scale_img.astype(np.uint8)

def segment_foreground(img_path, rect):
    import numpy as np
    import cv2

    ###
    # OpenCV uses BGR as its default colour order for images, matplotlib uses RGB.
    # When you display an image loaded with OpenCv in matplotlib the channels will be back to front.
    # The easiest way of fixing this is to use OpenCV to explicitly convert it back to RGB
    ###
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    height, width = img.shape[:2]
    if not rect:
        rect = (10, 0, width - 20, height - 10)
    else:
        [x, y, x1, y1] = list(rect)
        x2 = x + x1
        y2 = y + y1
        if(y >= height) or (x >= width) or (x2 <= 0) or (y2 <=0):
            return False, 0 , 0
        if (x < 0):
            x = 0
        if (x > width):
            x = width
        if (y < 0):
            y = 0
        if (y > height):
            y = height
        rect = (x,y,x1,y1)  # (start_x, start_y, width, height)

    _ = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)  # 3 iterations
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_black = img*mask2[:, :, np.newaxis]

    ## Black background to white background
    #Get the background
    background = img - img_black
    #Change all pixels in the background that are not black to white
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [
        255, 255, 255]
    #Add the background and the image
    img_white = background + img_black
    return True, img_white, mask2

def preprocessing_query(query_path, rect):
    base_query = 'static/query/'
    if (rect['width'] == 0 and rect['height'] == 0):
        rectangle = False
    else:
        rectangle = (int(rect['x']), int(rect['y']), int(rect['width']), int(rect['height']))
    validate, segmented, _ = segment_foreground(base_query + query_path, rectangle)
    if not validate :
        return False
    im = Image.fromarray(segmented)
    outname = 'static/after_query/' + os.path.basename(query_path)
    im.save(outname)
    return True

def preprocessing():
    img_list = sorted(glob.glob('data/*.jpg'))
    print(len(img_list))
    total = len(img_list)
    progress = 0

    for fname in img_list:
        # segmented is a segmented image on white background
        _, segmented, _ = segment_foreground(fname, False)

        if True:
            # save segmented image
            im = Image.fromarray(segmented)
            outname = 'after_preprocessing/' + os.path.basename(fname)
            im.save(outname)

            # report progress every 10 images completed
            progress = progress + 1

            if progress % 10 == 0:
                print('Progress: ' + str(progress/total))
                #print(outname)
