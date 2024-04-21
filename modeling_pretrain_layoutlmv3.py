import collections
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN, gelu
from transformers.modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    MaskedLMOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.layoutlmv3.configuration_layoutlmv3 import LayoutLMv3Config

###
from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3PreTrainedModel, LayoutLMv3Model, LayoutLMv3ClassificationHead, LayoutLMv3TextEmbeddings, LayoutLMv3PatchEmbeddings, LayoutLMv3Encoder
from transformers.models.layoutlmv3.modeling_layoutlmv3 import LAYOUTLMV3_DOWNSTREAM_INPUTS_DOCSTRING, LAYOUTLMV3_START_DOCSTRING, LAYOUTLMV3_MODEL_INPUTS_DOCSTRING
logger = logging.get_logger(__name__)

from transformers.models.roberta.modeling_roberta import RobertaLMHead

from modeling_output_layoutlmv3 import LayoutLMOutput, BaseLayoutLMOutput
import copy

_CONFIG_FOR_DOC = "LayoutLMv3Config"

@add_start_docstrings(
    "The bare LayoutLMv3 Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLMV3_START_DOCSTRING,
)

class BaseLayoutLMv3Model(LayoutLMv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.text_embed:
            self.embeddings = LayoutLMv3TextEmbeddings(config)

        if config.visual_embed:
            # use the default pre-training parameters for fine-tuning (e.g., input_size)
            # when the input_size is larger in fine-tuning, we will interpolate the position embeddings in forward
            self.patch_embed = LayoutLMv3PatchEmbeddings(config)

            size = int(config.input_size / config.patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.pos_embed = nn.Parameter(torch.zeros(1, size * size + 1, config.hidden_size))
            self.pos_drop = nn.Dropout(p=0.0)

            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                self.init_visual_bbox(image_size=(size, size))

            self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

            ### ml: mask_token (from beit)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.encoder = LayoutLMv3Encoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def init_visual_bbox(self, image_size=(14, 14), max_len=1000):
        """
        Create the bounding boxes for the visual (patch) tokens.
        """
        visual_bbox_x = torch.div(
            torch.arange(0, max_len * (image_size[1] + 1), max_len), image_size[1], rounding_mode="trunc"
        )
        visual_bbox_y = torch.div(
            torch.arange(0, max_len * (image_size[0] + 1), max_len), image_size[0], rounding_mode="trunc"
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_size[0], 1),
                visual_bbox_y[:-1].repeat(image_size[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_size[0], 1),
                visual_bbox_y[1:].repeat(image_size[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)

        cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
        self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)

    def calculate_visual_bbox(self, device, dtype, batch_size):
        visual_bbox = self.visual_bbox.repeat(batch_size, 1, 1)
        visual_bbox = visual_bbox.to(device).type(dtype)
        return visual_bbox

    def forward_image(self, pixel_values, bool_masked_pos=None):
        embeddings = self.patch_embed(pixel_values)
        
        ### ml: add image masking function
        batch_size, seq_len, _ = embeddings.size()
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        # add [CLS] token
        batch_size, seq_len, _ = embeddings.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add position embeddings
        if self.pos_embed is not None:
            embeddings = embeddings + self.pos_embed

        embeddings = self.pos_drop(embeddings)
        embeddings = self.norm(embeddings)

        return embeddings

    @add_start_docstrings_to_model_forward(
        LAYOUTLMV3_MODEL_INPUTS_DOCSTRING.format("batch_size, token_sequence_length")
    )
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        
        ### ml:
        im_mask: Optional[torch.LongTensor] = None,
        im_labels: Optional[torch.LongTensor] = None,

        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, words, boxes=boxes, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif pixel_values is not None:
            batch_size = len(pixel_values)
            device = pixel_values.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or pixel_values")

        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

        final_bbox = final_position_ids = None
        patch_height = patch_width = None
        if pixel_values is not None:
            patch_height, patch_width = (
                int(pixel_values.shape[2] / self.config.patch_size),
                int(pixel_values.shape[3] / self.config.patch_size),
            )
            ### ml: add image_mask
            visual_embeddings = self.forward_image(pixel_values, im_mask)

            visual_attention_mask = torch.ones(
                (batch_size, visual_embeddings.shape[1]), dtype=torch.long, device=device
            )
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            else:
                attention_mask = visual_attention_mask

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self.calculate_visual_bbox(device, dtype=torch.long, batch_size=batch_size)
                    if bbox is not None:
                        final_bbox = torch.cat([bbox, visual_bbox], dim=1)
                    else:
                        final_bbox = visual_bbox

                visual_position_ids = torch.arange(
                    0, visual_embeddings.shape[1], dtype=torch.long, device=device
                ).repeat(batch_size, 1)
                if input_ids is not None or inputs_embeds is not None:
                    position_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0)
                    position_ids = position_ids.expand(input_shape)
                    final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)
                else:
                    final_position_ids = visual_position_ids

            if input_ids is not None or inputs_embeds is not None:
                embedding_output = torch.cat([embedding_output, visual_embeddings], dim=1)
            else:
                embedding_output = visual_embeddings

            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, None, device, dtype=embedding_output.dtype
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_height=patch_height,
            patch_width=patch_width,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        
        ### ml: change output for visual embedding
        # return BaseModelOutput(
        return BaseLayoutLMOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            visual_embeddings=visual_embeddings
        )

class CustomRobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        ### ml: shrinking 2nd dim
        self.conv_1d = nn.Conv1d(in_channels=config.conv_input_size, out_channels=config.conv_output_size, kernel_size=1)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        ### ml: add 1d conv
        x = self.conv_1d(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias

class LayoutLMv3ForPretraining(LayoutLMv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlmv3 = BaseLayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lm_head = RobertaLMHead(config)
        # self.im_head = LayoutLMv3ClassificationHead(config, pool_feature=False)

        self.image_tokenizer = None 

        ### ml: add config.hidden_size, config.layer_norm_eps, config.vocab_size
        self.im_head_config = copy.deepcopy(config)
        self.im_head_config.conv_input_size = 197
        self.im_head_config.conv_output_size = 196
        self.im_head_config.vocab_size = 8192 ### ml: image_tokenizer vocab size
        self.im_head = CustomRobertaLMHead(self.im_head_config)

        ### TODO: CHANGE model type
        self.wpa_head_config = copy.deepcopy(config)
        self.wpa_head_config.conv_input_size = 709
        self.wpa_head_config.conv_output_size = 512
        self.wpa_head_config.vocab_size = 2 ### ml: alignment binary labels
        self.wpa_head = CustomRobertaLMHead(self.wpa_head_config)

        self.init_weights()

    
    @add_start_docstrings_to_model_forward(
        LAYOUTLMV3_DOWNSTREAM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        
        ### ml: custom params
        im_labels: Optional[torch.LongTensor] = None,
        im_mask:Optional[torch.LongTensor] = None,
        alignment_labels: Optional[bool] = None,

    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForTokenClassification
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]
        >>> word_labels = example["ner_tags"]

        >>> encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            
            ### ml: custom params
            im_labels=im_labels,
            im_mask=im_mask,
        )
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)

        im_sequence_output = outputs[0][:, seq_length:]
        im_sequence_output = self.dropout(im_sequence_output)

        wpa_sequence_output = outputs[0]
        wpa_sequence_output = self.dropout(wpa_sequence_output)

        ### ml: try to use LMHead and MLM loss
        prediction_scores = self.lm_head(sequence_output)

        def filter_values(matrix, filter_matrix):
            filtered_matrix = [[value for value, is_true in zip(row, filter_row) if is_true or is_true == 1] for row, filter_row in zip(matrix, filter_matrix)]
            return filtered_matrix

        mask_token = 50264
        masked_lm_loss = None
        
        ### TODO: convert to numpy array
        
        if labels is not None:
            # move labels to correct device to enable model parallelism
            lm_mask = [[value == mask_token for value in row] for row in input_ids.tolist()]
            
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            
            masked_labels = filter_values(labels, lm_mask)
            masked_prediction_scores = filter_values(prediction_scores, lm_mask)

            # masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            ### ml: flatten python list
            # flatten_masked_labels = torch.tensor([x for masked_label in masked_labels for x in masked_label]).to("cuda")
            flatten_masked_labels = torch.tensor([x for masked_label in masked_labels for x in masked_label]).to(self.device)
            flatten_masked_prediction_scores = torch.stack([x for masked_prediction_score in masked_prediction_scores for x in masked_prediction_score])
            masked_lm_loss = loss_fct(flatten_masked_prediction_scores, flatten_masked_labels)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        ### ml: im loss
        im_prediction_scores = self.im_head(im_sequence_output)
        masked_im_loss = None
        if im_labels is not None:
            # move labels to correct device to enable model parallelism
            im_labels = im_labels.to(im_prediction_scores.device)

            masked_im_labels = filter_values(im_labels, im_mask)
            masked_im_prediction_scores = filter_values(im_prediction_scores, im_mask)

            # flatten_masked_im_labels = torch.tensor([x for masked_label in masked_im_labels for x in masked_label]).to("cuda")
            flatten_masked_im_labels = torch.tensor([x for masked_label in masked_im_labels for x in masked_label]).to(self.device)
            flatten_masked_im_prediction_scores = torch.stack([x for masked_prediction_score in masked_im_prediction_scores for x in masked_prediction_score])

            loss_fct = CrossEntropyLoss()
            masked_im_loss = loss_fct(flatten_masked_im_prediction_scores,flatten_masked_im_labels)

        ### ml: wpa loss
        wpa_prediction_scores = self.wpa_head(wpa_sequence_output)
        masked_wpa_loss = None
        if alignment_labels is not None:
            # move labels to correct device to enable model parallelism
            alignment_labels = alignment_labels.to(wpa_prediction_scores.device)

            masked_wpa_labels = filter_values(alignment_labels, lm_mask)
            masked_wpa_prediction_scores = filter_values(wpa_prediction_scores, lm_mask)

            flatten_masked_wpa_labels = torch.tensor([x for masked_label in masked_wpa_labels for x in masked_label])
            flatten_masked_wpa_prediction_scores = torch.stack([x for masked_prediction_score in masked_wpa_prediction_scores for x in masked_prediction_score])

            ### ml: change wpa labels type (bool to nodes)
            _flatten_masked_wpa_labels = torch.zeros(flatten_masked_wpa_prediction_scores.shape)

            for i in range(flatten_masked_wpa_labels.size(0)):
                if flatten_masked_wpa_labels[i]:
                    _flatten_masked_wpa_labels[i, 0] = 1
                else:
                    _flatten_masked_wpa_labels[i, 1] = 1

            # flatten_masked_wpa_labels = _flatten_masked_wpa_labels.to("cuda")
            flatten_masked_wpa_labels = _flatten_masked_wpa_labels.to(self.device)

            loss_fct = CrossEntropyLoss()
            masked_wpa_loss = loss_fct(flatten_masked_wpa_prediction_scores, flatten_masked_wpa_labels)
            
        ### ml: mlm + mim + wpa loss
        ### TODO: apply cogview
        p_loss = masked_lm_loss + masked_im_loss + masked_wpa_loss
        
        return LayoutLMOutput(
            # loss=masked_lm_loss,
            loss=p_loss,
            logits=prediction_scores,
            reconstruction=im_labels,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

if __name__ == '__main__':
    from hf_utils.dataset.utils.funsd_utils import label_list, id2label, label2id
    model = LayoutLMv3ForPretraining.from_pretrained("microsoft/layoutlmv3-base",
                                                         id2label=id2label,
                                                         label2id=label2id)
    
    print('model loaded!')