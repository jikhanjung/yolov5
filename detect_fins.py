import imagesize
import os
import sys
import glob
from pathlib import Path
import csv
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import time

import argparse
import torch
from utils.general import strip_optimizer
from detect_mod import detect

from DolfinRecord import DolfinRecord, fieldnames

def get_image_info(filename):
    image_info = {'date':'','time':'','latitude':'','longitude':'','map_datum':''}
    i = Image.open(filename)
    ret = {}
    #print(filename)
    try:
        info = i._getexif()
        for tag, value in info.items():
            decoded=TAGS.get(tag, tag)
            ret[decoded]= value
            #print("exif:", decoded, value)
        try:
            if ret['GPSInfo'] != None:
                gps_info = ret['GPSInfo']
                #print("gps info:", gps_info)
            degree_symbol = "°"
            minute_symbol = "'"
            lon1, lon2, lon3 = gps_info[4][0], gps_info[4][1], gps_info[3]
            lat1, lat2, lat3 = gps_info[2][0], gps_info[2][1], gps_info[1]
            #print("lon1 type:", lon1, type(lon1).name)
            if isinstance(lon1,tuple):
                lon1 = int(lon1[0]) / int(lon1[1])
                lon2 = int(lon2[0]) / int(lon2[1])
                lat1 = int(lat1[0]) / int(lat1[1])
                lat2 = int(lat2[0]) / int(lat2[1])
                #lon3 = int(lon2[0]) / int(lon2[1])
            longitude = str(int(lon1)) + degree_symbol + str(lon2) + minute_symbol + lon3
            latitude = str(int(lat1)) + degree_symbol + str(lat2) + minute_symbol + lat3
            map_datum = gps_info[18]
            image_info['latitude'] = latitude
            image_info['longitude'] = longitude
            image_info['map_datum'] = map_datum

        except KeyError:
            print( "GPS Data Don't Exist for", Path(filename).name)

        try:
            if ret['DateTimeOriginal'] != None:
                exifTimestamp=ret['DateTimeOriginal']
                #print("original:", exifTimestamp)
                image_info['date'], image_info['time'] = exifTimestamp.split()
        except KeyError:
            print( "DateTimeOriginal Don't Exist")
        try:
            if ret['DateTimeDigitized'] != None:
                exifTimestamp= ret['DateTimeDigitized']
                image_info['date'], image_info['time'] = exifTimestamp.split()
        except KeyError:
            print( "DateTimeDigitized Don't Exist")
        try:
            if ret['DateTime'] != None:
                exifTimestamp= ret['DateTime']
                image_info['date'], image_info['time'] = exifTimestamp.split()
        except KeyError:
            print( "DateTime Don't Exist")

    except Exception as e:
        print(e)
    
    if image_info['date'] == '':
        str1 = time.ctime(os.path.getmtime(filename))
        datetime_object = datetime.strptime(str1, '%a %b %d %H:%M:%S %Y')
        image_info['date'] = datetime_object.strftime("%Y-%m-%d")
        image_info['time'] = datetime_object.strftime("%H:%M:%S")
    else:
        image_info['date'] = "-".join( image_info['date'].split(":") )
    image_info['datetime'] = image_info['date'] + ' ' + image_info['time']
    return image_info


def getOpt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='dolfin_1280_s_100.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', default=False, help='display results')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', default=False, help='update all models')
    opt = parser.parse_args()
    return opt

def detect_all_fins(folder_path, image_path_list):
    all_image_fin_list = []
    folder_name = folder_path.name
    obs_date, obs_location, obs_by = '', '', ''
    if "_" in folder_name:
        obs_list = folder_name.split("_",2)
        print(obs_list)
        if len(obs_list)==2:
            obs_date, obs_location = obs_list
        else:
            obs_date, obs_location, obs_by = obs_list
    else:
        obs_date = folder_name
        obs_location, obs_by = '', ''

    with torch.no_grad():
        all_result = detect(opt)

    for result in all_result:
        result.sort()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_image_fin_list = []

    i = 0
    for image_index in range(len(all_result)):
        single_image_result = all_result[image_index]

        image_info = get_image_info( str(image_path_list[image_index]))

        all_image_fin_list.append([])

        fin_count = len(single_image_result)
        if fin_count == 0:
            single_image_result = [ [ 0, 0, 0, 1, 1, -1 ] ]

        for fin_index in range(len(single_image_result)):
            #print( single_image_result[fin_index], image_info )
            cls, x, y, w, h, conf = single_image_result[fin_index]
            is_fin = True
            if conf < 0:
                is_fin = False
            fin_info = { 'folder_name': folder_name, 'image_name': Path( image_path_list[image_index] ).name,
                        'fin_index': fin_index+1, 'class_id': cls, 'center_x':x, 'center_y':y, 'width':w, 'height':h, 'confidence':conf, 
                        'is_fin': is_fin, 'image_datetime': image_info['datetime'], 'location': obs_location, 
                        'latitude':image_info['latitude'], 'longitude': image_info['longitude'], 'map_datum': image_info['map_datum'], 'dolfin_id':'', 
                        'observed_by': obs_by, 'created_by':'DolfinDetector_v0.0.1', 'created_on': now,
                        'modified_by': '', 'modified_on':'', 'comment': ''
                        }
            fin_record = DolfinRecord( fin_info )
            all_image_fin_list[image_index].append( fin_record )
    detection_done = True

    return all_image_fin_list
    
def save_data(folder_path, image_path_list, all_image_fin_list):
    folder_name = folder_path.name
    save_path = str( folder_path.joinpath( folder_name + ".csv" ))

    with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for image_index in range(len(image_path_list)):
            
            image_width, image_height = imagesize.get(str(image_path_list[image_index]))

            for fin_record in all_image_fin_list[image_index]:
                writer.writerow({'folder_name':fin_record.folder_name,'image_name': fin_record.image_name, 'image_width': image_width,
                                 'image_height': image_height,'class_id': int(fin_record.class_id), 
                                 'fin_index': fin_record.fin_index, 'center_x': fin_record.center_x, 'center_y': fin_record.center_y, 
                                 'width': fin_record.width, 'height': fin_record.height, 'confidence': fin_record.confidence,
                                 'is_fin': fin_record.is_fin, 'image_datetime': fin_record.image_datetime, 
                                 'location': fin_record.location, 'latitude': fin_record.latitude, 'longitude': fin_record.longitude,
                                 'map_datum': fin_record.map_datum, 'dolfin_id': fin_record.dolfin_id, 'observed_by': fin_record.observed_by, 
                                 'created_by': fin_record.created_by, 'created_on': fin_record.created_on,
                                 'modified_by': fin_record.modified_by, 'modified_on': fin_record.modified_on})
    return

def open_folder(folder_name):
    folder_path = Path(folder_name)

    image_path_list = []

    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

    p = folder_name
    p = os.path.abspath(p)  # absolute path
    
    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]

        for img in images:
            #print(img)
            image_path_list.append( img )
    return image_path_list


if __name__ == "__main__" :
    opt = getOpt()

    if opt.source == '':
        opt.source = "./20160316_TEST"
    folder_name = opt.source
    folder_path = Path(folder_name)

    image_path_list = open_folder(folder_path)

    all_image_fin_list = detect_all_fins(folder_path, image_path_list)

    save_data(folder_path, image_path_list, all_image_fin_list)
