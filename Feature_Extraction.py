import os
from matplotlib import pyplot as plt
import mysql.connector
import numpy as np
from skimage import feature
import cv2
from PIL import Image
def convert_to_rgb(image):
    """Chuyển đổi ảnh về dạng RGB nếu nó không phải là RGB."""
    if len(image.shape) == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        if image.shape[2] == 4:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            rgb_image = image
        else:
            raise ValueError("Không xác định định dạng ảnh đầu vào.")
    else:
        raise ValueError("Không xác định định dạng ảnh đầu vào.")
    
    return rgb_image


def my_calcHist(image, channels, histSize, ranges):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], channels, None, histSize, ranges)
    hist /= (hsv_image.shape[0] * hsv_image.shape[1])
    hist = hist / hist.max()
    return hist

def convert_image_rgb_to_gray(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    return img_clahe

def hog_feature(gray_img):
    (hog_feats, hogImage) = feature.hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2",
        visualize=True,
    )

    # hog_feats /= np.linalg.norm(hog_feats)
    # hog_feats = hog_feats / hog_feats.max()

    
    return hog_feats, hogImage

def edge_feature(image):
    
    edges = cv2.Canny(image, 100, 200,
                 apertureSize = 5,  
                 L2gradient = True)
    return edges.flatten()/255

def layout_feature(image):
    image = image /255
    grid_size = 10
    height, width = image.shape
    cell_height, cell_width = height // grid_size, width // grid_size
    layout_features = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = image[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
            cell_feature = np.mean(cell)
            layout_features.append(cell_feature)
    return np.array(layout_features)

def feature_extraction(img):
    channels = [0, 1, 2]
    histSize = [32, 32, 24]
    ranges = [0, 180, 0, 256, 0, 256]
    hist = my_calcHist(img, channels, histSize, ranges)
    embedding = np.array(hist.flatten()) / (256*256)
    gray_image = convert_image_rgb_to_gray(img)
    embedding_hog, hog_img = list(hog_feature(gray_image))
    
    # Edge detection
    edge_feat = edge_feature(gray_image)
    
    # Layout analysis
    layout_feat = layout_feature(gray_image)
    
    return embedding, embedding_hog, hog_img, edge_feat, layout_feat

