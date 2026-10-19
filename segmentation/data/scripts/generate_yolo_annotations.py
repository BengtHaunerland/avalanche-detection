import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict

#########
# Constants
#########

GLIDE = 'glide'
LOOSE = 'loose'
SLAB = 'slab'
NONE = 'none'

IDX_TO_CLASS: Dict[int, str] = {0: GLIDE, 1: LOOSE, 2: SLAB}
CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}
VALID_PROB = 0.1  # Proportion of training images in validation set

LABELS = ['glide', 'loose', 'none', 'slab']


def _convert_label_file(output_path: str = None, annotation_path: str = None):
    """
    Convert a VOTT annotation file into the format expected by YOLO.

    Args:
        output_path (str): the output path of the converted annotation file. 
        annotation_path (str): the path of the original VOTT annotation file for the image.

    Returns:
        None.
    """
    # Load annotation object in VOTT format
    with open(annotation_path, 'rb') as f:
        ann_obj: Dict = json.load(f)

    img_size = ann_obj['asset']['size']
    img_size['height'], img_size['width']

    seg_s = []

    for region in ann_obj['regions']:
        if region['type'] != 'POLYGON':
            raise ValueError('Unsupported region type ' + region['type'])

        assert len(
            region['tags']) <= 1, f'Bounding box has multiple labels for {annotation_path}'

        class_idx = CLASS_TO_IDX[region['tags'][0]]
        points = region['points']
        if len(points) < 3:
            raise ValueError(f'Unsupported polygon with fewer than 3 points in {annotation_path}')

        segment = [str(class_idx)]
        for point in points:
            segment.append(str(point['x'] / img_size['width']))
            segment.append(str(point['y'] / img_size['height']))

        seg_s.append(segment)

    # Write converted annotation to label file
    with open(output_path, 'w') as f:
        for seg in seg_s:
            f.write(' '.join(seg) + '\n')


def convert_annotation_files(data_dir: str = '.data'):
    '''
    Convert annotation files into the format expected by YOLO.

    Args:
        data_dir (str): the directory containing the images and annotations.

    Returns:
        None.
    '''
    # Create empty annotation folders
    ann_train_dir = Path(data_dir) / 'train' / 'labels'
    ann_test_dir = Path(data_dir) / 'test' / 'labels'

    for folder in [ann_train_dir, ann_test_dir]:
        # Delete folder if it exists
        if folder.exists():
            shutil.rmtree(folder)
        for l in LABELS:
            label_folder = folder / l
            label_folder.mkdir(parents=True, exist_ok=False)

    for im_dir in [Path(data_dir) / 'train' / 'images', Path(data_dir) / 'test' / 'images']:
        for root, _, filenames in os.walk(im_dir):
            root_path = Path(root)
            for file_name in filenames:
                im_name, file_ext = os.path.splitext(file_name)
                assert file_ext == '.jpg', f'Bad file extension {file_name}'

                # Copy image into destination directory
                im_path = root_path / file_name
                label = im_path.parent.name

                assert label in LABELS, f'Unknown label {label}'

                # Get annotation path from image path for non-NONE images and convert to YOLO format
                if label != NONE:
                    ann_path = Path(data_dir) / 'annotations' / label / f'{im_name}.json'
                    output_path = root_path.parent.parent / 'labels' / label / f'{im_name}.txt'
                    _convert_label_file(output_path=str(output_path),
                                        annotation_path=str(ann_path))


def create_train_valid_split(train_valid_seed: int = None, data_dir: str = '.data', dest_dir: str = '.data/yolo_split'):
    '''Create text files with paths to images in the train/test/valid test splits

    Args:
        train_valid_seed (int): seed for the split into train/validation sets.
        data_dir (str): the directory containing the images and annotations.
        dest_dir (str): the directory to save generated split files.

    Returns:
        None.
    '''
    test_dir = f'{data_dir}/test/images'
    train_dir = f'{data_dir}/train/images'

    # Set random seed for train/valid split
    random.seed(train_valid_seed)

    # Create txt files for all images in the test and train image directories
    for crawl_dir in [test_dir, train_dir]:
        for root, _, filenames in os.walk(crawl_dir):
            for file_name in filenames:
                if not file_name.lower().endswith('.jpg'):
                    continue

                # Split train images into validation set with probability VALID_PROB
                if crawl_dir == test_dir:
                    split_file = 'test'
                else:
                    split_file = 'val' if random.random() < VALID_PROB else 'train'
                txt_path = os.path.join(
                    dest_dir, f'{split_file}{train_valid_seed}.txt')

                # Write image path to txt file
                f_txt = open(txt_path, 'a')
                abs_path = os.path.abspath(os.path.join(root, file_name))
                f_txt.write(abs_path + '\n')
                f_txt.close()

    print(f'Txt files created with seed {train_valid_seed}')


if __name__ == '__main__':
    # Check that data is downloaded
    DATA_DIR = '.data'
    ANNOTATIONS_DIR = f'{DATA_DIR}/annotations'
    for dir in [DATA_DIR, f'{DATA_DIR}/train/images',  f'{DATA_DIR}/test/images', ANNOTATIONS_DIR]:
        assert os.path.isdir(
            dir), f'Directory not found: {os.path.abspath(dir)}'

    # Create annotation files in format expected by YOLO
    convert_annotation_files(data_dir=DATA_DIR)

    # Create experiment_dir if needed
    train_val_split_dir = os.path.join(DATA_DIR, 'yolo_split')
    try:
        shutil.rmtree(train_val_split_dir)
    except:
        pass
    os.makedirs(train_val_split_dir, exist_ok=False)

    # Create txt files for test/train/validation splits with different "random" seeds
    for valid_seed in [50, 100, 150]:
        create_train_valid_split(
            valid_seed, data_dir=DATA_DIR, dest_dir=train_val_split_dir)
