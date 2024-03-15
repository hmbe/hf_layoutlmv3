import sys, os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

from datasets import load_dataset
from PIL import Image
# from transformers import LayoutLMv2Processor, LayoutXLMProcessor, LayoutLMv3Processor
from processing_pretrain_layoutlmv3 import LayoutLMv3PretrainProcessor

from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from hf_utils.dataset.utils.funsd_utils import label_list, id2label, label2id

from datasets import load_metric
import numpy as np

from transformers import LayoutLMv3ForTokenClassification
from modeling_pretrain_layoutlmv3 import LayoutLMv3ForPretraining

from transformers import AutoConfig, AutoModel

from transformers import TrainingArguments, Trainer
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

from utils_layoutlmv3 import create_alignment_label, init_visual_bbox

from data_collator_pretrain_layoutlmv3 import DataCollatorForLayoutLMv3
import evaluate
import torch

example_dataset = load_dataset("nielsr/funsd-layoutlmv3", streaming=True)
model = LayoutLMv3ForPretraining.from_pretrained("microsoft/layoutlmv3-base",
                                                         id2label=id2label,
                                                         label2id=label2id)

auto_config = AutoConfig.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3PretrainProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        # return torch.tensor(mask.flatten())

        ### ml: convert 0, 1 to True, False 
        return torch.tensor(mask.flatten()).to(torch.bool)

mask_generator = MaskGenerator(
    input_size = auto_config.input_size,
    mask_patch_size = auto_config.patch_size * 2,
    model_patch_size = auto_config.patch_size,
)

def prepare_examples(examples, is_train=True):
    images = examples['image']
    words = examples['tokens']
    boxes = examples['bboxes']
    word_labels = examples['ner_tags']

    encoding = processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, stride =128,
        padding="max_length", max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True)

    ### im_mask
    encoding["im_mask"] = [mask_generator() for i in range(len(encoding['pixel_values']))]

    ### visual_bbox
    text_bboxes = encoding['bbox']
    image_poses = encoding['im_mask']
    visual_bbox = init_visual_bbox()

    ### batch 별 별도 처리 수행
    encoding["alignment_labels"] = []
    for batch_idx in range(len(text_bboxes)):
        text_bbox = text_bboxes[batch_idx]
        image_pos = image_poses[batch_idx]
        encoding["alignment_labels"].append(create_alignment_label(visual_bbox, text_bbox, image_pos)) 

    offset_mapping = encoding.pop('offset_mapping')
    overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')

    return encoding

# we need to define custom features for `set_format` (used later on) to work properly
features = Features({
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(feature=Value(dtype='int64')),
    'im_labels': Sequence(feature=Value(dtype='int64')),
    'im_mask': Sequence(feature=Value(dtype='int64')),
    'alignment_labels': Sequence(feature=Value(dtype='bool')),
})

### join data collator for applying mlm
# pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
data_collator = DataCollatorForLayoutLMv3(
    tokenizer=processor.tokenizer,
    mlm_probability=0.9,
    pad_to_multiple_of=None,
)

# Set the training transforms
# example_dataset["train"].set_transform(preprocess_images)

column_names = example_dataset["train"].column_names
train_dataset = example_dataset["train"].map(
    prepare_examples,
    batched=True,
    batch_size=16,
    remove_columns=column_names,
    features=features,
)

test_dataset = example_dataset["test"].map(
    prepare_examples,
    batched=True,
    batch_size=16,
    remove_columns=column_names,
    features=features,
)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

metric = evaluate.load("accuracy")

### TODO: add mim accuracy logic
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels[0].reshape(-1)
    ### ml: add argmax
    preds = preds[0].argmax(axis=-1).reshape(-1)
    
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


training_args = TrainingArguments(output_dir="test",
                                  max_steps=5000,
                                  per_device_train_batch_size=4,
                                  per_device_eval_batch_size=4,
                                  learning_rate=1e-5,
                                  evaluation_strategy="steps",
                                  eval_steps=100,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="eval_loss",
                                  save_strategy="steps",
                                  save_steps=500,
                                  save_total_limit=5,
                                  greater_is_better=False,
                                  overwrite_output_dir=True,
                                  )


# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()