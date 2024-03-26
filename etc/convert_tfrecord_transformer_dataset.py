import torch
from tfrecord.torch.dataset import TFRecordDataset

tfrecord_path = "D:\\github\\test.tfrecord"
index_path = None
# description = {"image": "byte", "label": "float"}

feature_description = {
    'pixel_values': "float",
    'input_ids': "int",
    'attention_mask': "int",
    'bbox': "int",
    'labels': "int",
    'im_labels': "int",
    'im_mask': "int",
    'alignment_labels': "int",
}

# feature_description = {
#     'pixel_values': tf.io.FixedLenFeature(dtype=tf.float, shape=(3, 224, 224)),
#     'input_ids': tf.io.FixedLenFeature([], dtype=tf.int64),
#     'attention_mask': tf.io.FixedLenFeature([], dtype=tf.int64),
#     'bbox': tf.io.FixedLenFeature(dtype=tf.int64, shape=(512, 4)),
#     'labels': tf.io.FixedLenFeature([], dtype=tf.int64),
#     'im_labels': tf.io.FixedLenFeature([], dtype=tf.int64),
#     'im_mask': tf.io.FixedLenFeature([], dtype=tf.int64),
#     'alignment_labels': tf.io.FixedLenFeature([], dtype=tf.int64),
# }

dataset = TFRecordDataset(tfrecord_path, index_path, feature_description)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

data = next(iter(loader))
print(data)