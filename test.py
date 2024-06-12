import os
from matplotlib import pyplot as plt
import mysql.connector
import numpy as np
from skimage import feature
import cv2
from PIL import Image
import math
from Feature_Extraction import feature_extraction, convert_to_rgb


def calculate_pearson_correlation(vec1, vec2):
    return np.corrcoef(vec1, vec2)[0, 1]

def combined_similarity(features_1, features_2, weights):
    similarity = 0
    for f1, f2, w in zip(features_1, features_2, weights):
        sim = calculate_pearson_correlation(f1, f2)
        similarity += w * sim
    return similarity
def image_distance(color_hist1, hog1, edge1, layout1, color_hist2, hog2, edge2, layout2):
    features_1 = [color_hist1, hog1, edge1, layout1]
    features_2 = [color_hist2, hog2, edge2, layout2]
    weights = [0.4, 0.4, 0.1, 0.1 ]
    similarity = combined_similarity(features_1, features_2, weights)
    return similarity


def search_folder(imageinput):
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="natural_images"
    )
    cursor = db.cursor()
    cursor.execute("SELECT foldername, color_histogram, hog, edges, layout FROM averagedata")
    rows = cursor.fetchall()
    
    color_hist_1, hog_1, _, edge_1, layout_1 = feature_extraction(imageinput)
    distances = []
    for row in rows:
        filename, color_hist_str, hog_str, edge_str, layout_str = row
        color_hist = np.array(list(map(float, color_hist_str.split(','))))
        hog = np.array(list(map(float, hog_str.split(','))))
        edge = np.array(list(map(float, edge_str.split(','))))
        layout = np.array(list(map(float, layout_str.split(','))))
        print(filename)
        distance = image_distance(color_hist_1, hog_1, edge_1, layout_1, color_hist, hog, edge, layout)
        distances.append((distance, filename))
    
    distances.sort(key=lambda x: x[0], reverse=True)
    closest_image_filenames = [filename for _, filename in distances[:3]]
    foldername = closest_image_filenames[0]
    db.close()
    return foldername, distances
def search_image(imageinput, outputfolder):
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="natural_images"
    )
    cursor = db.cursor()
    cursor.execute(f"SELECT filenametest, color_histogram, hog, edges, layout FROM {outputfolder}" )
    rows = cursor.fetchall()
    
    color_hist_1, hog_1, _, edge_1, layout_1 = feature_extraction(imageinput)
    distances = []
    for row in rows:
        filename, color_hist_str, hog_str, edge_str, layout_str = row
        color_hist = np.array(list(map(float, color_hist_str.split(','))))
        hog = np.array(list(map(float, hog_str.split(','))))
        edge = np.array(list(map(float, edge_str.split(','))))
        layout = np.array(list(map(float, layout_str.split(','))))
        print(filename)
        distance = image_distance(color_hist_1, hog_1, edge_1, layout_1, color_hist, hog, edge, layout)
        distances.append((distance, filename))
    
    distances.sort(key=lambda x: x[0], reverse=True)
    closest_image_filenames = [filename for _, filename in distances[:3]]
    
    db.close()
    return closest_image_filenames, distances

def test():
    imageinput = cv2.imread(r"C:\Users\Admin\Downloads\Search Natural Images\test1.png")
    rgb_image = convert_to_rgb(imageinput)
    output_folder, dis_out_folder = search_folder(rgb_image)
    print("Danh sách các nhãn và độ tương đồng", dis_out_folder)
    print("Ảnh thuộc loại ", output_folder)

    output_folder = str(output_folder)
    output, dis_out = search_image(rgb_image, output_folder)
    print("Danh sách các Ảnh thuộc nhãn và độ tương đồng", dis_out)
    print("Các ảnh giống nhất là: ", output)
    
    image_folder = r"C:\Users\Admin\Downloads\Search Natural Images\dataset\\" + str(output_folder)
    
    images = []
    for filename in output:
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        images.append(image)
    

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title("Ảnh đầu vào")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    plt.title("Ảnh tương tự 1")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
    plt.title("Ảnh tương tự 2")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB))
    plt.title("Ảnh tương tự 3")
    plt.axis('off')


    plt.show()

test()
