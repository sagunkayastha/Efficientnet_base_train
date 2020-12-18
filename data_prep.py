import os
from random import shuffle
import shutil
import splitfolders

full_data_path = 'Data_full'
balanced_path = 'balanced_dataset'


def balance():

    for folder in os.listdir(full_data_path):

        # Make folder in balanced
        os.makedirs(os.path.join(balanced_path,folder),exist_ok=True)

        fold = os.listdir(os.path.join(full_data_path,folder))
        shuffle(fold)
        if len(fold)>2500:
            fold = fold[:2500]

        for img in fold:
            src = os.path.join(full_data_path,folder,img)
            dest = os.path.join(balanced_path,folder,img)
            shutil.copy(src,dest)
# balance()
splitfolders.ratio(balanced_path, output="data_split", seed=1337, ratio=(.8, .1, .1), group_prefix=None)


 for folder in os.listdir(full_data_path):

        # Make folder in balanced
        os.makedirs(os.path.join(balanced_path,folder),exist_ok=True)

        fold = os.listdir(os.path.join(full_data_path,folder))
        shuffle(fold)
        if len(fold)>2500:
            fold = fold[:2500]

        for img in fold:
            src = os.path.join(full_data_path,folder,img)
            dest = os.path.join(balanced_path,folder,img)
            shutil.copy(src,dest)