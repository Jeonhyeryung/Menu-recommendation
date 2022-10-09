# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy

import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    bm25_weight,
)

from modules import Encoder, LayerNorm, activation_layer, MLPLayers
from utils import mf_sgd, get_predicted_full_matrix, get_rmse, item_encoding, als, get_ALS_loss


class BERT4RecModel(nn.Module):
    def __init__(self, args):
        super(BERT4RecModel, self).__init__()
        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.out = nn.Linear(args.hidden_size, args.item_size - 1)
        self.args = args

        #self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence, args):

        seq_length = sequence.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)

        if not args.rm_position:
            position_embeddings = self.position_embeddings(position_ids)
            sequence_emb = item_embeddings + position_embeddings
        else:
            sequence_emb = item_embeddings

        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).repeat(1, input_ids.shape[1], 1).unsqueeze(1).long()

        if self.args.cuda_condition:
            extended_attention_mask = extended_attention_mask.cuda()

        # extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids, self.args)

        item_encoded_layers = self.item_encoder(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )

        sequence_output = item_encoded_layers[-1]
        sequence_output = self.out(sequence_output)

        return sequence_output

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
