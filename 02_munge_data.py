"""
The purpose of this python script is to create an unbiased training and validation set.
The split data will be run in the terminal calling a function (process_data) that will join the
annotations.csv file with new .txt files for bounding box class and coordinates for each image.
"""
# Credit to Abhishek Thakur, as this is a modified version of this notebook.
# Source to video, where he goes over his code: https://www.youtube.com/watch?v=NU9Xr_NYslo&t=1392s

# Import libraries
import os
import ast
import pandas as pd
import numpy as np
from sklearn import model_selection
from tqdm import tqdm
import shutil

# The DATA_PATH will be where your augmented images and annotations.csv files are.
# The OUTPUT_PATH is where the train and validation images and labels will go to.
DATA_PATH = '/home/dlee/Documents/git-personal/GA_Data_Science_Capstone/asl_modeling_data/aug_data/'
OUTPUT_PATH = '/home/dlee/Documents/git-personal/GA_Data_Science_Capstone/yolov5/asl_yolo/'

# Function for taking each row in the annotations file
def process_data(data, data_type='train'):
    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_name = row['image_id'][:-4]  # removing file extension .jpeg
        bounding_boxes = row['bboxes']
        yolo_data = []
        for bbox in bounding_boxes:
            category = bbox[0]
            x_center = bbox[1]
            y_center = bbox[2]
            w = bbox[3]
            h = bbox[4]
            yolo_data.append([category, x_center, y_center, w, h]) # yolo formated labels
        yolo_data = np.array(yolo_data)

        np.savetxt(
            # Outputting .txt file to appropriate train/validation folders
            os.path.join(OUTPUT_PATH, f"labels/{data_type}/{image_name}.txt"),
            yolo_data,
            fmt=["%d", "%f", "%f", "%f","%f"]
        )
        shutil.copyfile(
            # Copying the augmented images to the appropriate train/validation folders
            os.path.join(DATA_PATH, f"images/{image_name}.jpg"),
            os.path.join(OUTPUT_PATH, f"images/{data_type}/{image_name}.jpg"),
        )

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(DATA_PATH, 'annotations.csv'))
    df.bbox = df.bbox.apply(ast.literal_eval) # Convert string to list for bounding boxes
    df = df.groupby('image_id')['bbox'].apply(list).reset_index(name='bboxes')

    # splitting data to a 90/10 split
    df_train, df_valid = model_selection.train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    shuffle=True
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # Run function to have our data ready for modeling in 03_Modeling_and_Inference.ipynb
    process_data(df_train, data_type='train')
    process_data(df_valid, data_type='validation')
