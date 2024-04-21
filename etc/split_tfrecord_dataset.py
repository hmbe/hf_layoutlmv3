import tensorflow as tf
import os, sys
tfrecord_input_file = '/home/mingi.lim/workspace/hf_layoutlmv3/data/test01/test.tfrecord'
tfrecord_output_dir = '/home/mingi.lim/workspace/hf_layoutlmv3/data/test01_split/'

def read_tfrecord(serialized_example):
    feature_description = {
        # 'pixel_values': tf.io.FixedLenFeature(dtype=tf.float32, shape=(3, 224, 224)),
        'pixel_values': tf.io.FixedLenFeature([], dtype=tf.float32),
        'input_ids': tf.io.FixedLenFeature([], dtype=tf.int64),
        'attention_mask': tf.io.FixedLenFeature([], dtype=tf.int64),
        'bbox': tf.io.FixedLenFeature([], dtype=tf.int64),
        'im_labels': tf.io.FixedLenFeature([], dtype=tf.int64),
        'im_mask': tf.io.FixedLenFeature([], dtype=tf.int64),
        'alignment_labels': tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)
    return example

raw_dataset = tf.data.TFRecordDataset(tfrecord_input_file)
raw_dataset = raw_dataset.map(read_tfrecord)

def write_tfrecord(file_name, dataset):
    writer = tf.io.TFRecordWriter(file_name)
    for record in dataset:
        writer.write(record.SerializeToString())
    writer.close()

it = iter(raw_dataset)
split_num = 10
batch_size = 16
count = 0

try:
    while True:
        current_dataset = []
        for _ in range(batch_size):
            current_dataset.append(next(it))

        output_file_name = f"{tfrecord_output_dir}{os.path.splitext(os.path.basename(tfrecord_input_file))[0]}-{count}.tfrecord"

        temp_ds = tf.data.Dataset.from_tensor_slices(current_dataset)
        write_tfrecord(output_file_name)

        print(f'output file name: {output_file_name}')
        print(f'current dataset: {current_dataset}')

        count += 1
        count = count % batch_size

except StopIteration:
    if current_dataset:
        output_file_name = f"{tfrecord_output_dir}{os.path.splitext(os.path.basename(tfrecord_input_file))[0]}-{count}.tfrecord"
        temp_ds = tf.data.Dataset.from_tensor_slices(current_dataset)
        write_tfrecord(output_file_name)