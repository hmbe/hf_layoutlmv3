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
import cv2

from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
import torch
import requests

from dall_e.encoder import Encoder
from dall_e.decoder import Decoder

import json
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3Tokenizer, LayoutLMv3TokenizerFast, BertTokenizer

### tensorflow dependencies
import tensorflow as tf

from raw_dataset_generator_layoutlmv3 import AihubRawDataset
import threading

ROOT_DIR='/mnt/nas-drive-workspace/Datasets/aihub/'

FILE_FORMAT = 'tfrecord'
SAVE_DIR = '/mnt/nas-drive-workspace/Datasets/aihub-preprocessed/023_OCR_DATA_PUBLIC/'
TARGET_DATASET_TYPES = ['train', 'validation']
DATASET_SPLIT_NUMS = [10, 4]

TARGET_AIHUB_DATASET=['023_OCR_DATA_PUBLIC']

### 1. load dataset
# dataset = load_dataset("nielsr/funsd-layoutlmv3", streaming=True)
# dataset = load_dataset("./raw_dataset_generator_layoutlmv3.py", target_aihub_datasets=['023_OCR_DATA_PUBLIC'], root_dir='/data/aihub/', streaming=True)

### 2. load processor
def load_image_tokenizer(path='./dall_e_tokenizer/encoder.pkl'):
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path)
        resp.raise_for_status()
            
        with io.BytesIO(resp.content) as buf:
            return torch.load(buf)
    elif os.path.splitext(path)[1] == '.pkl':
        if path == './dall_e_tokenizer/encoder.pkl':
            if not os.path.exists(path):
                print(f'{path} is not exist! download dall-e encoder.pkl now..')
                resp = requests.get('https://cdn.openai.com/dall-e/encoder.pkl')
                resp.raise_for_status()
                os.makedirs(os.path.dirname(path), exist_ok=True)

                with open(path, "wb") as file:
                    file.write(resp.content)

        with open(path, 'rb') as f:
            return torch.load(f)
    
    ### ml: custom image tokenizer
    elif os.path.splitext(path)[1] == '.pt':
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        vae_model = torch.load(path, device)
        enc = Encoder()
        enc_state_dict = {}
        for key, value in vae_model['module'].items():
            if 'vae.enc.' == key[:len('vae.enc.')]:
                enc_state_dict[key[len('vae.enc.'):]] = value
            # elif 'vae.dec.' == key[:len('vae.dec.')]:
            #     dec_state_dict[key[len('vae.dec.'):]] = value
                
        enc.load_state_dict(enc_state_dict, device)
        return enc
           
# image_tokenizer = load_image_tokenizer('/home/mingi.lim/workspace/dall_e_tokenizer/global_step608626_19M_ep1.x_dalle_v2/model_states.pt')
# image_tokenizer = load_image_tokenizer()
image_tokenizer = load_image_tokenizer('/mnt/nas-drive-workspace/Models/tokenizer/dall_e_tokenizer/global_step608626_19M_ep1.x_dalle_v2/model_states.pt')
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")

image_processor_config_str = """{
  "apply_ocr": false,
  "do_normalize": true,
  "do_resize": true,
  "feature_extractor_type": "LayoutLMv3FeatureExtractor",
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "ocr_lang": null,
  "resample": 2,
  "size": 224
}"""

image_processor_kwargs = json.loads(image_processor_config_str)
image_processor = LayoutLMv3ImageProcessor(**image_processor_kwargs)

# processor = LayoutLMv3PretrainProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
processor = LayoutLMv3PretrainProcessor(image_processor=image_processor, tokenizer=tokenizer, image_tokenizer=image_tokenizer)

### 3. set mask generator
auto_config = AutoConfig.from_pretrained("microsoft/layoutlmv3-base")
mask_generator = MaskGenerator(
    input_size = auto_config.input_size,
    mask_patch_size = auto_config.patch_size * 2,
    model_patch_size = auto_config.patch_size,
)

### 4. make features
def prepare_encoding(examples):
    if 'image' in examples.keys():
        images = examples['image']
    elif 'image_path' in examples.keys():
        images = cv2.imread(examples['image_path'])

    if 'tokens' in examples.keys():
        words = examples['tokens']
    elif 'words' in examples.keys():
        words = examples['words']

    boxes = examples['bboxes']

    if 'ner_tags' in examples.keys():
        word_labels = examples['ner_tags']
    else:
        word_labels = None
    
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
        # 'labels': tf.io.FixedLenFeature([], dtype=tf.int64),
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
        # 'labels': Sequence(feature=Value(dtype='int64')),
        'im_labels': Sequence(feature=Value(dtype='int64')),
        'im_mask': Sequence(feature=Value(dtype='int64')),
        'alignment_labels': Sequence(feature=Value(dtype='bool')),
    })

### TODO: fix logic for loading data

def write_tfrecord(thread_id, file_name, encodings):
    with tf.io.TFRecordWriter(file_name) as writer:
        for encoding in tqdm(encodings):
            encoding = prepare_encoding(encoding)
            for i in range(len(encoding['input_ids'])):
                example = tf.train.Example(features=tf.train.Features(feature={
                    'pixel_values': tf.train.Feature(float_list=tf.train.FloatList(value=encoding['pixel_values'][i].flatten())),
                    'input_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=encoding['input_ids'][i])),
                    'attention_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=encoding['attention_mask'][i])),
                    'bbox': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(encoding['bbox'][i]).flatten())),
                    # 'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=encoding['labels'][i])),
                    'im_labels': tf.train.Feature(int64_list=tf.train.Int64List(value=encoding['im_labels'][i].numpy())),
                    'im_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=encoding['im_mask'][i].numpy())),
                    'alignment_labels': tf.train.Feature(int64_list=tf.train.Int64List(value=encoding['alignment_labels'][i].numpy())),
                }))

                writer.write(example.SerializeToString())

print('start creating dataset!')
for i, dataset_type in enumerate(TARGET_DATASET_TYPES):
    threads = []
    dataset_split_num = DATASET_SPLIT_NUMS[i]
    for j in range(dataset_split_num):
        file_name = f'{"_".join(TARGET_AIHUB_DATASET)}_{dataset_type}_{j}.{FILE_FORMAT}'
        dataset = load_dataset("./raw_dataset_generator_layoutlmv3.py", 
                                target_aihub_datasets=TARGET_AIHUB_DATASET, 
                                root_dir=ROOT_DIR,
                                dataset_split_num = dataset_split_num,
                                dataset_split_idx = j,
                                streaming=True)
        encodings = dataset[dataset_type]
        thread = threading.Thread(target=write_tfrecord, args=(j, file_name, encodings))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
