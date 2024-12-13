import dataclasses
import enum

import polars as pl
import torch
from omegaconf import DictConfig
from torch import nn
from transformers import AutoModel, AutoTokenizer

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.utils.module_class import Module


@dataclasses.dataclass
class ModelOutput:
    rep: torch.Tensor
    hidden_states: torch.Tensor = None


class Triplet(enum.Enum):
    DATE = "date"
    VALUE = "value"
    VARIABLE = "variable"


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


class CVE(nn.Module):
    """Continuous Value Encoder (CVE) module.

    Assumes input is a single continuous value, and encodes it as an `output_dim` size embedding vector.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Linear(1, cfg.token_dim)

    def forward(self, x):
        return self.layer(x)


class TextCodeEmbedder(nn.Module, Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.code_to_tokens_map = self.build_code_to_tokens_map()
        self.code_embedder = AutoModel.from_pretrained(self.cfg.code_embedder)
        self.linear = nn.Linear(self.code_embedder.config.hidden_size, self.cfg.token_dim)

    def build_code_to_tokens_map(self):
        """
        Builds a mapping from code to tokens

        Returns:
            code_to_tokens_map: A dictionary mapping from code to tokens

        """
        code_metadata = pl.scan_parquet(self.cfg.code_metadata_fp).select(["code/vocab_index", "description"])
        code_metadata = code_metadata.sort("code/vocab_index").collect()
        # check that there is no 0 -- this should be reserved for the padding token
        assert (
            code_metadata.select(["code/vocab_index"]).min().item() == 1
        ), "Vocab index should start from 1."
        # check that there is no missing index
        assert (
            code_metadata.select(["code/vocab_index"]).max().item() == code_metadata.shape[0]
        ), "Vocab index should be continuous."

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.code_tokenizer)
        tokenized_code_metadata = tokenizer(
            ["[PAD]"] + code_metadata.select(["description"]).fill_null("").to_series().to_list(),
            **self.cfg.tokenizer_config,
        )
        return tokenized_code_metadata

    def forward(self, codes, mask):
        unique_codes = codes.unique()
        sorted_unique_codes, indices = torch.sort(unique_codes)
        relative_codes = torch.searchsorted(sorted_unique_codes, codes)
        # relative_codes corresponds to the indices of the sorted unique codes
        # so that we can just use the output of the embedder without realigning

        # sorted_unique_codes = sorted_unique_codes.to(codes.device)
        # relative_codes = relative_codes.to(codes.device)
        for key in self.code_to_tokens_map.keys():
            self.code_to_tokens_map[key] = self.code_to_tokens_map[key].to(codes.device)

        embedder_inputs = {
            key: self.code_to_tokens_map[key][sorted_unique_codes].to(codes.device)
            for key in self.code_to_tokens_map.keys()
        }
        code_embeddings = self.code_embedder(**embedder_inputs).pooler_output

        code_embeddings = self.linear(code_embeddings)

        embeddings = code_embeddings[relative_codes].to(codes.device)

        # TODO: Masking -- not sure if this is needed as we can do it later
        return embeddings


class TextCodeEncoder(nn.Module, Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.

    Copied from: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Define Triplet Embedders
        self.date_embedder = CVE(cfg)
        self.code_embedder = TextCodeEmbedder(cfg)
        self.numeric_value_embedder = CVE(cfg)

    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].transpose(2, 0)).permute(1, 2, 0)
        return out

    def get_embedding(self, batch):
        static_mask = batch["static_mask"]
        code = batch["code"]
        code_mask = batch["mask"]
        numeric_value = batch["numeric_value"]
        time_delta_days = batch["time_delta_days"]
        numeric_value_mask = batch["numeric_value_mask"]
        # Embed times and mask static value times
        time_emb = self.embed_func(self.date_embedder, time_delta_days) * ~static_mask.unsqueeze(dim=1)
        # Embed codes
        code_emb = self.code_embedder.forward(code, code_mask)
        code_emb = code_emb.permute(0, 2, 1)

        # Embed numerical values and mask nan values
        val_emb = self.embed_func(self.numeric_value_embedder, numeric_value) * numeric_value_mask.unsqueeze(
            dim=1
        )

        # Sum the (time, code, value) triplets and
        embedding = time_emb + code_emb + val_emb

        assert embedding.isfinite().all(), "Embedding is not finite"
        if embedding.shape[-1] > self.cfg.max_seq_len:
            raise ValueError(
                f"Triplet embedding length {embedding.shape[-1]} "
                "is greater than max_seq_len {self.cfg.max_seq_len}"
            )
        return embedding.transpose(1, 2)

    def forward(self, batch):
        embedding = self.get_embedding(batch)
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        batch[INPUT_ENCODER_TOKENS_KEY] = embedding
        return batch
