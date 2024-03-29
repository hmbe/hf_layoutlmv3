# import pickle5 as pickle
import sys
import os
import numpy as np
import pyarrow.parquet as pq
import pandas as pd 
from tqdm import tqdm

from datasets import load_dataset
from processing_pretrain_layoutlmv3 import LayoutLMv3PretrainProcessor

sys.path.append('../src')
# from utils import utils, masking_generator

from utils_layoutlmv3 import create_alignment_label, init_visual_bbox
from utils_layoutlmv3 import MaskGenerator
from transformers import AutoConfig, AutoModel

import pandas as pd
import numpy as np
from PIL import Image
import io
import pyarrow as pa
import pyarrow.parquet as pq

from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

### tensorflow dependencies
import tensorflow as tf

FILE_FORMAT = 'tfrecord'
FILE_NAME = 'test'
    
### 1. load dataset
### column_names: ['id', 'tokens', 'bboxes', 'ner_tags', 'image']
### 변경 column_names: ['pixel_values', 'input_ids', ]
### 추가 column_names: ['im_mask', 'alignment_labels', 'im_labels']
### 다른 데이터셋도 본 column name 따르기
example_dataset = load_dataset("nielsr/funsd-layoutlmv3", streaming=True)

### 2. load processor
processor = LayoutLMv3PretrainProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

### 3. set mask generator
auto_config = AutoConfig.from_pretrained("microsoft/layoutlmv3-base")
mask_generator = MaskGenerator(
    input_size = auto_config.input_size,
    mask_patch_size = auto_config.patch_size * 2,
    model_patch_size = auto_config.patch_size,
)

### 4. make features
def prepare_encoding(examples):
    images = examples['image']
    words = examples['tokens']
    boxes = examples['bboxes']
    word_labels = examples['ner_tags']
    
    ### im_labels
    encoding = processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, stride=128, padding="max_length", max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True)

    ### im_mask
    encoding["im_mask"] = [mask_generator() for i in range(len(encoding['pixel_values']))]

    ### visual_bbox
    text_bboxes = encoding['bbox']
    image_poses = encoding['im_mask']
    visual_bbox = init_visual_bbox()

    encoding["alignment_labels"] = []
    for batch_idx in range(len(text_bboxes)):
        text_bbox = text_bboxes[batch_idx]
        image_pos = image_poses[batch_idx]
        encoding["alignment_labels"].append(create_alignment_label(visual_bbox, text_bbox, image_pos, is_bool=False)) 

    offset_mapping = encoding.pop('offset_mapping')
    overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')

    return encoding

### 5. make feature description
if FILE_FORMAT == 'tfrecord':
    feature_description = {
        'pixel_values': tf.io.FixedLenFeature(dtype=tf.float32, shape=(3, 224, 224)),
        'input_ids': tf.io.FixedLenFeature([], dtype=tf.int64),
        'attention_mask': tf.io.FixedLenFeature([], dtype=tf.int64),
        'bbox': tf.io.FixedLenFeature(dtype=tf.int64, shape=(512, 4)),
        'labels': tf.io.FixedLenFeature([], dtype=tf.int64),
        'im_labels': tf.io.FixedLenFeature([], dtype=tf.int64),
        'im_mask': tf.io.FixedLenFeature([], dtype=tf.int64),
        'alignment_labels': tf.io.FixedLenFeature([], dtype=tf.int64),
    }

elif FILE_FORMAT == 'parquet':
    feature_description = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(feature=Value(dtype='int64')),
        'im_labels': Sequence(feature=Value(dtype='int64')),
        'im_mask': Sequence(feature=Value(dtype='int64')),
        'alignment_labels': Sequence(feature=Value(dtype='bool')),
    })

### TODO: fix logic for loading data
encodings = example_dataset['train']

if FILE_FORMAT == 'tfrecord':
    # TFRecordWriter를 사용하여 TFRecord 파일 생성
    with tf.io.TFRecordWriter(f'{FILE_NAME}.{FILE_FORMAT}') as writer:
        for encoding in tqdm(encodings):
            encoding = prepare_encoding(encoding)
            for i in range(len(encoding['input_ids'])):
                example = tf.train.Example(features=tf.train.Features(feature={
                    'pixel_values': tf.train.Feature(float_list=tf.train.FloatList(value=encoding['pixel_values'][i].flatten())),
                    'input_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=encoding['input_ids'][i])),
                    'attention_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=encoding['attention_mask'][i])),
                    'bbox': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(encoding['bbox'][i]).flatten())),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=encoding['labels'][i])),
                    'im_labels': tf.train.Feature(int64_list=tf.train.Int64List(value=encoding['im_labels'][i].numpy())),
                    'im_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=encoding['im_mask'][i].numpy())),
                    'alignment_labels': tf.train.Feature(int64_list=tf.train.Int64List(value=encoding['alignment_labels'][i].numpy())),
                }))

                writer.write(example.SerializeToString())