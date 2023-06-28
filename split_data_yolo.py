import argparse
import glob
from sklearn.model_selection import train_test_split
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--folder_in', type=str)
args = parser.parse_args()

all_images_paths = glob.glob(f'{args.folder_in}/*.jpg')
train, valtest = train_test_split(all_images_paths, test_size = 0.2, random_state=42)
val, test = train_test_split(valtest, test_size = 0.5, random_state=42)
for path in train:
    txt_path = path.replace(".jpg", ".txt")
    shutil.move(path,f'./dataset/train/images/{path.split("/")[-1]}')
    shutil.move(path,f'./dataset/train/labels/{txt_path.split("/")[-1]}')

for path in val:
    txt_path = path.replace(".jpg", ".txt")
    shutil.move(path,f'./dataset/val/images/{path.split("/")[-1]}')
    shutil.move(path,f'./dataset/val/labels/{txt_path.split("/")[-1]}')

for path in test:
    shutil.move(path,f'./dataset/train/images/{path.split("/")[-1]}')