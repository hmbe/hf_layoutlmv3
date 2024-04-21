# coding=utf-8
import json
import os
import glob

import datasets

from PIL import Image
import numpy as np

from create_filelist_layoutlmv3 import get_file_paths

logger = datasets.logging.get_logger(__name__)

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

class AihubConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(AihubConfig, self).__init__(**kwargs)

class AihubRawDataset(datasets.GeneratorBasedBuilder):
    """AIHUB Raw dataset."""

    BUILDER_CONFIGS = [
        AihubConfig(name="aihub", version=datasets.Version("1.0.0"), description="AIHUB dataset"),
    ]

    def __init__(self, *args, **kwargs):
        self.target_aihub_datasets = kwargs.pop('target_aihub_datasets', [])
        self.root_dir = kwargs.pop('root_dir', None)

        self.dataset_split_num = kwargs.pop('dataset_split_num', 1)
        self.dataset_split_idx = kwargs.pop('dataset_split_idx', 0)
        super().__init__(*args, **kwargs)
        
        # self.target_aihub_datasets = ['023_OCR_DATA_PUBLIC', '032_PUBLIC_ADMIN_DOCUMENT_OCR', '025_OCR_DATA_FINANCE_LOGISTICS']
        # self.target_aihub_datasets = ['023_OCR_DATA_PUBLIC']
        # self.target_aihub_datasets = ['032_PUBLIC_ADMIN_DOCUMENT_OCR']
        
    def _info(self):
        return datasets.DatasetInfo(
            description=None,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"root_dir": f"{self.root_dir}", "target_datasets": self.target_aihub_datasets, "type": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"root_dir": f"{self.root_dir}", "target_datasets": self.target_aihub_datasets, "type": "validation"}
            ),
        ]
    
    def _generate_examples(self, root_dir, target_datasets, type):
        ### generate full file path with type and target_dataset
        ### get paths of img_files and ann_files
        print(target_datasets)
        for dataset_name in target_datasets:
            ### make ann_dir and img_dir for each datasets
            # logger.info(f"⏳ Generating examples.. root_dir: {root_dir}, target_datasets: {target_datasets}")
            print(f"⏳ Generating examples.. root_dir: {root_dir}, target_datasets: {target_datasets}")

            img_files, ann_files = get_file_paths(root_dir, dataset_name, type)
            # self.dataset_split_num = kwargs.pop('dataset_split_num', 1)
            # self.dataset_split_idx = kwargs.pop('dataset_split_idx', 0)

            quotient = float(len(img_files)) / float(self.dataset_split_num)
            img_files = img_files[int(quotient*self.dataset_split_idx):int(quotient*(self.dataset_split_idx+1))]
            ann_files = ann_files[int(quotient*self.dataset_split_idx):int(quotient*(self.dataset_split_idx+1))]

            for guid, (img_file, ann_file) in enumerate(zip(img_files, ann_files)):
                words = []
                bboxes = []
                try:
                    with open(ann_file, "r", encoding="utf8") as f:
                        data = json.load(f)
                except:
                    print(f'{ann_file} can not be loaded!')
                    continue
                
                try:
                    image, size = load_image(img_file)
                except:
                    print(f'{img_file} can not be loaded!')
                    continue
                
                if dataset_name  == '023_OCR_DATA_PUBLIC':
                    json_bboxes = data['Bbox']
                    size = (data['Images']['width'], data['Images']['height'])
                    # word : str, bbox : -> xyxy, size : wh
                    for json_bbox in json_bboxes:
                        word = json_bbox['data']
                        bbox = normalize_bbox([min(json_bbox['x']), min(json_bbox['y']), max(json_bbox['x']), max(json_bbox['y'])], size)
                        words.append(word)
                        bboxes.append(bbox)

                elif dataset_name == '032_PUBLIC_ADMIN_DOCUMENT_OCR':
                    json_bboxes = data['bbox']
                    size = (data['Images']['width'], data['Images']['height'])
                    # word : str, bbox : -> xyxy, size : wh
                    for json_bbox in json_bboxes:
                        word = json_bbox['data']
                        bbox = normalize_bbox([min(json_bbox['x']), min(json_bbox['y']), max(json_bbox['x']), max(json_bbox['y'])], size)
                        words.append(word)
                        bboxes.append(bbox)

                elif dataset_name == '025_OCR_DATA_FINANCE_LOGISTICS':
                    json_bboxes = data['bbox']
                    size = (data['Images']['width'], data['Images']['height'])
                    # word : str, bbox : -> xyxy, size : wh
                    for json_bbox in json_bboxes:
                        word = json_bbox['data']
                        bbox = normalize_bbox([min(json_bbox['x']), min(json_bbox['y']), max(json_bbox['x']), max(json_bbox['y'])], size)
                        words.append(word)
                        bboxes.append(bbox)

                elif dataset_name == '055_FINANCE_OCR_DATA':
                    json_polygons = data['annotations'][0]['polygons']
                    size = (data['images'][0]['width'], data['images'][0]['height'])

                    # word : str, bbox : -> xyxy, size : wh
                    for i, json_polygon in enumerate(json_polygons):
                        word = json_polygon['text']
                        json_polygon_x_list = [p[0] for p in json_polygon['points']]
                        json_polygon_y_list = [p[1] for p in json_polygon['points']]

                        bbox = normalize_bbox([min(json_polygon_x_list), min(json_polygon_y_list), max(json_polygon_x_list), max(json_polygon_y_list)], size)
                        words.append(word)
                        bboxes.append(bbox)
                    
                yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "image_path": img_file}


if __name__ == '__main__':
    target_aihub_datasets = ['055_FINANCE_OCR_DATA']
    root_dir = '/mnt/nas-drive-workspace/Datasets/aihub/'
    aihub_raw_builder = AihubRawDataset(root_dir=root_dir, target_aihub_datasets=target_aihub_datasets)
    aihub_raw_builder.download_and_prepare()
    dataset = aihub_raw_builder.as_dataset(split='train')
    print(dataset)
    print(next(iter(dataset)))
    print('done!')