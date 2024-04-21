# coding=utf-8
import json
import os
import glob

import datasets

from PIL import Image
import numpy as np

import tfrecord
from tfrecord.reader import tfrecord_loader
from tfrecord.torch.dataset import TFRecordDataset

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
        self.target_aihub_datasets = kwargs.pop('target_aihub_datasets', [])

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

        if os.path.splitext(self.target_path)[1] == '.tfrecord':
            print('load tfrecord preprocessed dataset!')

            # feature_description = {
            #     'pixel_values': tf.io.FixedLenFeature(dtype=tf.float32, shape=(3, 224, 224)),
            #     'input_ids': tf.io.FixedLenFeature([], dtype=tf.int64),
            #     'attention_mask': tf.io.FixedLenFeature([], dtype=tf.int64),
            #     'bbox': tf.io.FixedLenFeature(dtype=tf.int64, shape=(512, 4)),
            #     'im_labels': tf.io.FixedLenFeature([], dtype=tf.int64),
            #     'im_mask': tf.io.FixedLenFeature([], dtype=tf.int64),
            #     'alignment_labels': tf.io.FixedLenFeature([], dtype=tf.int64),
            # }

            index_path = None
            feature_description = {
                'pixel_values': "float",
                'input_ids': "int",
                'attention_mask': "int",
                'bbox': "int",
                'im_labels': "int",
                'im_mask': "int",
                'alignment_labels': "int",
            }

            loader = tfrecord_loader(self.target_path, index_path, feature_description)

        for guid, record in enumerate(loader):
            yield guid, record


if __name__ == '__main__':
    # aihub_preprocessed_builder = AihubPreprocessedDataset(target_path='/home/mingi.lim/workspace/_test.tfrecord')
    aihub_preprocessed_builder = AihubPreprocessedDataset(target_path='/home/mingi.lim/workspace/test.tfrecord')
    
    aihub_preprocessed_builder.download_and_prepare()
    dataset = aihub_preprocessed_builder.as_dataset(split='train')
    print(dataset)
    print(next(iter(dataset)))

    print('done!')