# coding=utf-8
import json
import os
import glob

import datasets

from PIL import Image
import numpy as np

from tfrecord.reader import tfrecord_loader
import tfrecord

from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

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

class AihubPreprocessedDataset(datasets.GeneratorBasedBuilder):
    """AIHUB Preprocessed dataset."""

    BUILDER_CONFIGS = [
        AihubConfig(name="aihub-preprocessed", version=datasets.Version("1.0.0"), description="AIHUB dataset"),
    ]

    def __init__(self, *args, **kwargs):
        # self.target_aihub_datasets = kwargs.pop('target_aihub_datasets', [])
        self.target_path = kwargs.pop('target_path', None)

        # we need to define custom features for `set_format` (used later on) to work properly
        self.feature_description = Features({
            'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(feature=Value(dtype='int64')),
            'im_labels': Sequence(feature=Value(dtype='int64')),
            'im_mask': Sequence(feature=Value(dtype='int64')),
            'alignment_labels': Sequence(feature=Value(dtype='bool')),
        })

        super().__init__(*args, **kwargs)
        
        # self.target_aihub_datasets = ['023_OCR_DATA_PUBLIC', '032_PUBLIC_ADMIN_DOCUMENT_OCR', '025_OCR_DATA_FINANCE_LOGISTICS']
        # self.target_aihub_datasets = ['023_OCR_DATA_PUBLIC']
        # self.target_aihub_datasets = ['032_PUBLIC_ADMIN_DOCUMENT_OCR']
        

    def _info(self):
        return datasets.DatasetInfo(
            description=None,
            features=datasets.Features(
                {
                    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
                    'input_ids': Sequence(feature=Value(dtype='int64')),
                    'attention_mask': Sequence(Value(dtype='int64')),
                    'bbox': Array2D(dtype="int64", shape=(512, 4)),
                    # 'labels': Sequence(feature=Value(dtype='int64')),
                    'im_labels': Sequence(feature=Value(dtype='int64')),
                    'im_mask': Sequence(feature=Value(dtype='int64')),
                    'alignment_labels': Sequence(feature=Value(dtype='bool')),
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
                name=datasets.Split.TRAIN, gen_kwargs={"type": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"type": "validation"}
            ),
        ]
    
    def _generate_examples(self, type):
        logger.info(f"⏳ Generating examples.. target_path: {self.target_path}")
        ### load preprocessed dataset
        if len(os.path.splitext(self.target_path)) == 1:
            print('target_path is directory!')
        if os.path.splitext(self.target_path) == '.tfrecord':
            print('load tfrecord preprocessed dataset!')
            
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
                
            yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "image_path": img_file}


if __name__ == '__main__':
    aihub_preprocessed_builder = AihubPreprocessedDataset(target_path='/home/mingi.lim/workspace/_test.tfrecord')
    aihub_preprocessed_builder.download_and_prepare()
    dataset = aihub_preprocessed_builder.as_dataset(split='train')
    print(dataset)
    print(next(iter(dataset)))

    print('done!')