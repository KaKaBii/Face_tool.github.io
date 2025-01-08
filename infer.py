import os
import argparse
import yaml
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt

from detector.detect import detect_face
from detector.smooth import smooth_face
from utils.image import (load_image, 
                         save_image, 
                         save_steps, 
                         check_img_size,
                         get_height_and_width,
                         process_image,
                         check_if_adding_bboxes)
from utils.video import (split_video,
                         process_video)
from utils.types import (is_image,
                         is_video,
                         is_directory)


def parse_args():
    """
    Argument parser for cli.

    Returns
    -------
    args : ArgumentParser object
        Contains all the cli arguments
    """
    parser = argparse.ArgumentParser(description='Facial detection and \
                                     smoothing using OpenCV.')
    parser.add_argument('--input', 
                        type=str, 
                        help='Input file or folder',
                        default='data/images/01.jpg')
    parser.add_argument('--output', 
                        type=str, 
                        help='Output file or folder',
                        default='data/output')
    parser.add_argument('--show-detections', 
                        action='store_true',
                        help='Displays bounding boxes during inference.')
    parser.add_argument('--save-steps', 
                        action='store_true',
                        help='Saves each step of the image.')
    args = parser.parse_args()
    # assert args.image_shape is None or len(args.image_shape) == 2, \
    #     'You need to provide a 2-dimensional tuple as shape (H,W)'
    # assert (is_image(args.input) and is_image(args.output)) or \
    #        (not is_image(args.input) and not is_image(args.input)), \
    #     'Input and output must both be images or folders'
    return args


def load_configs():
    """
    Loads the project configurations.

    Returns
    -------
    configs : dict
        A dictionary containing the configs
    """
    with open('configs/configs.yaml', 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def clear_old_output_files(output_dir):
    """清除 'output_[數值].jpg' 檔案，數值小於30"""
    for filename in os.listdir("./data/output"):
        if filename.startswith('Output_') :
            try:
                os.remove(os.path.join(output_dir, filename))
                print(f"已刪除檔案: {filename}")
            except ValueError:
                continue


def main(args):
    """Puts it all together."""
    # Start measuring time
    tic = time.perf_counter()
    # Load project configurations
    cfg = load_configs()
    # Load the network
    net = cv2.dnn.readNetFromTensorflow(cfg['net']['model_file'], 
                                        cfg['net']['cfg_file'])
    # 輸出資料夾清除舊檔案
    clear_old_output_files(args.output)
    
    # 開始處理檔案，處理01~30的圖片
    for i in range(1, 31):
        # 格式化檔名（例如01.jpg, 02.jpeg, ...）
        file_name = f"{i}"  # 將數字格式化為兩位數字（01, 02, 03...）
        image_formats = ['jpg', 'jpeg', 'webp']  # 支援的圖片格式
        found_image = False

        # 檢查並處理每個格式的圖片
        for ext in image_formats:
            input_file = os.path.join('data/images', f"{i}.{ext}")
            if os.path.exists(input_file):  # 檢查檔案是否存在
                found_image = True
                break  # 如果找到檔案則跳出內層for迴圈

        if not found_image:
            print(f"File {i} with supported format not found, skipping.")
            continue

        try:
            # 如果是影片檔案
            if is_video(input_file):
                process_video(input_file, args, cfg, net)
            
                      # 如果是圖片檔案
            elif is_image(input_file):
                input_img = load_image(input_file)  # 載入圖片
                img_steps = process_image(input_img, cfg, net)  # 處理圖片

                # 設定輸出檔案名稱，使用 "Output_" 作為檔名前綴
                output_file_name = f"Output_{i}"  # 輸出格式化檔名
                out_filename = os.path.join(args.output, output_file_name)
                
                # 如果需要顯示邊界框則加入邊界框
                output_img = check_if_adding_bboxes(args, img_steps)
                img_saved = save_image(out_filename, output_img)  # 儲存圖片
            
        except ValueError:
            print(f"Input {input_file} is not valid.")
        
 
    # End measuring time
    toc = time.perf_counter()
    print(f"Operation ran in {toc - tic:0.4f} seconds")
    # Save processing steps
    if args.save_steps:
        # Set image output height
        output_height = cfg['image']['img_steps_height']
        # Set output filename
        steps_filename = os.path.join(args.output, cfg['image']['output_steps'])
        # Save file
        save_steps(steps_filename, img_steps, output_height)

    # End measuring time
    toc = time.perf_counter()
    print(f"Operation ran in {toc - tic:0.4f} seconds")


if __name__ == '__main__':
    args = parse_args()
    main(args)
