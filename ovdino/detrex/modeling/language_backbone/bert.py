import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertConfig, BertModel

logger = logging.getLogger(__name__)


class BERTTokenizer:
    """BERT tokenizer.
    Args:
        tokenizer_name (str): the name of tokenizer.
        padding_mode (int): the padding mode of tokenizer.
            'max_length': padding the tokenined token to the max_length.
            'longest': padding the tokenized token to the length of the longest token.
        contex_length (int): the max context_length of tokenizer
            if given padding_mode is 'max_length'. Default is 'longest'.
        bos_token_id (int): the bos_token_id of tokenizer.
        eos_token_id (int): the eos_token_id of tokenizer.
        pad_token_id (int): the pad_token_id of tokenizer.
        dot_token_id (int): the dot_token_id of tokenizer.
    Example:
    """

    def __init__(
        self,
        tokenizer_name="bert-base-uncased",
        padding_mode="longest",
        context_length=48,
        dot_token_id=1012,
        bos_token_id=101,
        eos_token_id=102,
        pad_token_id=0,
    ) -> None:
        super().__init__()
        # BERT tokenizer in Huggingface, the bos_token_id and eos_token_id is None, we need to define them here.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.padding_mode = padding_mode
        self.context_length = context_length
        self.dot_token_id = dot_token_id
        self.bos_token_id = (
            bos_token_id
            if self.tokenizer.bos_token_id is None
            else self.tokenizer.bos_token_id
        )
        self.eos_token_id = (
            eos_token_id
            if self.tokenizer.eos_token_id is None
            else self.tokenizer.eos_token_id
        )
        self.pad_token_id = (
            pad_token_id
            if self.tokenizer.pad_token_id is None
            else self.tokenizer.pad_token_id
        )

    def __call__(self, x, return_mask=False, *args: Any, **kwargs: Any) -> Any:
        tokenized_batch = self.tokenizer.batch_encode_plus(
            x,
            *args,
            max_length=self.context_length,
            padding=self.padding_mode,
            return_special_tokens_mask=True,
            return_tensors="pt",
            truncation=True,
            **kwargs,
        )
        output = {"input_ids": tokenized_batch["input_ids"]}
        if return_mask:
            output["attention_mask"] = tokenized_batch["attention_mask"]
        return output


class BERTEncoder(nn.Module):
    def __init__(
        self,
        tokenizer_cfg=dict(tokenizer_name="bert-base-uncased"),
        model_name="bert-base-uncased",
        output_dim=256,
        padding_mode="longest",
        context_length=48,
        pooling_mode="max",
        post_tokenize=False,
        is_normalize=False,
        is_proj=False,
        is_freeze=False,
        return_dict=False,
    ) -> None:
        super().__init__()
        assert pooling_mode in ["max", "mean", None]
        self.bos_token_id = 101
        self.eos_token_id = 102
        self.padding_mode = padding_mode
        self.context_length = context_length
        self.post_tokenize = post_tokenize
        self.tokenizer = BERTTokenizer(**tokenizer_cfg)
        lang_model_config = BertConfig.from_pretrained(model_name)
        self.lang_model = BertModel.from_pretrained(
            model_name, add_pooling_layer=False, config=lang_model_config
        )
        self.is_normalize = is_normalize
        self.pooling_mode = pooling_mode
        self.is_proj = is_proj
        self.is_freeze = is_freeze
        self.return_dict = return_dict
        self.num_layers = 1
        if self.is_proj:
            self.text_porj = nn.Parameter(
                torch.empty(self.lang_model.config.hidden_size, output_dim)
            )
            nn.init.normal_(
                self.text_porj, std=self.lang_model.config.hidden_size**-0.5
            )
            # self.text_porj = nn.Linear(self.lang_model.config.hidden_size, output_dim)
        # freeze parameters
        if self.is_freeze:
            logger.info(f"Freezee parameters of {self.__class__.__name__}.")
            for param in self.lang_model.parameters():
                param.requires_grad = False

    def forward(self, x, *args, **kwargs):
        """Forward function of text_encoder.
        Args:
            x (list[str] or Tensor): the input text that is a list of category name(or definition) or toneized token.
                shape: N x [C, ] or [N, C, L]([N, L]), where L is the context_length.
        Returns:
            output (Tensor): the extracted text feature, shape: [N*C, D].
        """
        if self.post_tokenize:
            tokenized_batch = self.tokenizer(x, return_mask=True)
            input_ids = tokenized_batch["input_ids"].cuda()
            attention_mask = tokenized_batch["attention_mask"].cuda()
        else:
            assert self.context_length == x.shape[-1]
            input_ids = x.reshape(-1, self.context_length)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        output = self.lang_model(
            input_ids=input_ids, output_hidden_states=True, *args, **kwargs
        )["last_hidden_state"]
        output_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "last_hidden_state": output,
        }
        if self.pooling_mode == "mean":
            output = torch.mean(output, dim=1, keepdim=False)
            output_dict.update({"pooled_output": output})
        elif self.pooling_mode == "max":
            # take features from the eos_token embedding
            eot_indices = torch.nonzero(
                torch.eq(input_ids, self.tokenizer.eos_token_id)
            )
            output = output[torch.arange(output.shape[0]), eot_indices[:, 1]]
            output_dict.update({"pooled_output": output})
        else:
            raise NotImplementedError("Only support pooling_mode: [max, mean].")

        if self.is_normalize:
            output = F.normalize(output, p=2, dim=-1)
            output_dict.update({"normalized_output": output})
        if self.is_proj:
            output = output @ self.text_porj
            output_dict.update({"projected_output": output})
        if self.return_dict:
            return output_dict
        return output
