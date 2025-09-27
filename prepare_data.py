import os
import csv
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

#making new directories
os.makedirs('dataset/train', exist_ok=True)
os.makedirs('dataset/test', exist_ok=True)
os.makedirs('dataset/val', exist_ok=True)

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

with open('fer2013.csv') as f:
    data = csv.reader(f)
    next(data)

    for i, row in enumerate(data):
        emotion_idx = int(row[0])
        emotion = emotions[emotion_idx]
        usage = row[2]
        pixels = np.array([int(p) for p in row[1].split()], dtype=np.uint8)
        img = pixels.reshape(48, 48)

        #making subdirectories for each emotion
        if usage == 'Training':
            path = f'dataset/train/{emotion}'
        elif usage == 'PublicTest':
            path = f'dataset/val/{emotion}'
        else:
            path = f'dataset/test/{emotion}'

        os.makedirs(path, exist_ok=True)
        cv2.imwrite(f'{path}/img_{i}.png', img)

print("Dataset preparation complete!")