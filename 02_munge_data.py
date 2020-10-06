
import os
import ast
import pandas as pd
import numpy as np
from sklearn import model_selection
from tqdm import tqdm
import shutil

DATA_PATH = '/home/dlee/Documents/git-personal/GA_Data_Science_Capstone/asl_modeling_data/aug_data/'
OUTPUT_PATH = '/home/dlee/Documents/git-personal/GA_Data_Science_Capstone/yolov5/asl_yolo/'

def process_data(data, data_type='train'):
    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_name = row['image_id'][:-4]
        bounding_boxes = row['bboxes']
        yolo_data = []
        for bbox in bounding_boxes:
            category = bbox[0]
            x_center = bbox[1]
            y_center = bbox[2]
            w = bbox[3]
            h = bbox[4]
            yolo_data.append([category, x_center, y_center, w, h])
        yolo_data = np.array(yolo_data)
        # print(yolo_data)
        np.savetxt(
            os.path.join(OUTPUT_PATH, f"labels/{data_type}/{image_name}.txt"),
            yolo_data,
            fmt=["%d", "%f", "%f", "%f","%f"]
        )
        shutil.copyfile(
            os.path.join(DATA_PATH, f"images/{image_name}.jpg"),
            os.path.join(OUTPUT_PATH, f"images/{data_type}/{image_name}.jpg"),
        )

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(DATA_PATH, 'annotations.csv'))
    df.bbox = df.bbox.apply(ast.literal_eval) # Convert string to list for bounding boxes
    df = df.groupby('image_id')['bbox'].apply(list).reset_index(name='bboxes')

    df_train, df_valid = model_selection.train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    shuffle=True
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    process_data(df_train, data_type='train')
    process_data(df_valid, data_type='validation')
