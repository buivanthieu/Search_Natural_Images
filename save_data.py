import os
import mysql.connector
import numpy as np
from skimage import feature
import cv2
from PIL import Image
from Feature_Extraction import feature_extraction, convert_to_rgb
def read_image(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and any(filename.endswith(ext) for ext in ['.png']):
            image = cv2.imread(file_path)
            rgb_image = convert_to_rgb(image)
            if image is not None:
                images.append((filename, rgb_image))
    return images

def connect_to_database():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="natural_images",
        ssl_disabled=True,
        connection_timeout=600, # Đặt thời gian chờ lâu hơn
    )

def save_to_database(images):
    db = connect_to_database()
    print("connect thành công")

    cursor = db.cursor()
    cursor.execute(f"DELETE FROM {naturalfolderdata}")
    # cursor.execute("set global max_allowed_packet=67108864;")
    print("Xóa dữ liệu cũ thành công")
    for filename, image in images:
        embedding, embedding_hog, _, edge_feature, layout_feature = feature_extraction(image)
        print(edge_feature.shape, layout_feature.shape)
        color_hist = ','.join(map(str, embedding))
        hog = ','.join(map(str, embedding_hog))
        edges = ','.join(map(str, edge_feature))
        layout = ','.join(map(str, layout_feature))
        sql = f"INSERT INTO {naturalfolderdata} (filenametest, color_histogram, hog, edges, layout) VALUES (%s, %s, %s, %s, %s)"
        val = (filename, color_hist, hog, edges, layout)
        
        cursor.execute(sql, val)
        print("đang lưu trữ data %s", naturalfolderdata)
    print("completed %s", naturalfolderdata)
    db.commit()
    db.close()
def average_vector(listvector):
    return sum(listvector) / len(listvector)
def average_data():
    db = connect_to_database()
    cursor = db.cursor()
    cursor.execute(f"SELECT filenametest, color_histogram, hog, edges, layout FROM {naturalfolderdata}")
    rows = cursor.fetchall()
    color_hist_list = []
    hog_list  = []
    edge_list  = []
    layout_list  =[]
    for row in rows:
        _, color_hist_str, hog_str, edge_str, layout_str = row
        color_hist = np.array(list(map(float, color_hist_str.split(','))))
        hog = np.array(list(map(float, hog_str.split(','))))
        edge = np.array(list(map(float, edge_str.split(','))))
        layout = np.array(list(map(float, layout_str.split(','))))
        color_hist_list.append(color_hist)
        hog_list.append(hog)
        edge_list.append(edge)
        layout_list.append(layout)
    color_hist_average = average_vector(color_hist_list)
    hog_list_average = average_vector(hog_list)
    edge_list_average = average_vector(edge_list)
    layout_list_average = average_vector(layout_list)
    return color_hist_average, hog_list_average, edge_list_average, layout_list_average

def save_average_data_to_database(na):
    db = connect_to_database()
    print("connect thành công")
    color_hist_average, hog_list_average, edge_list_average, layout_list_average = average_data()
    cursor = db.cursor()
    color_hist = ','.join(map(str, color_hist_average))
    hog = ','.join(map(str, hog_list_average))
    edges = ','.join(map(str, edge_list_average))
    layout = ','.join(map(str, layout_list_average))
    sql = "INSERT INTO averagedata (foldername, color_histogram, hog, edges, layout) VALUES (%s, %s, %s, %s, %s)"
    val = (na,color_hist, hog, edges, layout)
    cursor.execute(sql, val)
    db.commit()
    db.close()

folderlist = ["bien", "nuithac", "samac", "thaonguyen", "songho"]
for naturalfolderdata in folderlist:
    folder_path = r"C:\Users\Admin\Downloads\Search Natural Images\dataset\\" + str(naturalfolderdata)
    images = read_image(folder_path)
    save_to_database(images)
    save_average_data_to_database(naturalfolderdata)

