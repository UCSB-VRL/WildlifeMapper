# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        aux_loss=False,
        embed_dim=256
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.aux_loss = aux_loss
        self.num_classes = 8 + 1 #id 9 is reserved for background and id "0" is not used, total 10

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        #hfc cross attention
        # self.hfc_attn = CrossAttentionHfcPrompt(
        #     d_model = embed_dim,
        #     nhead = 8,
        #     dropout = 0.1,
        #     dim_feedforward = embed_dim,
        #     activation = 'relu',
        #     proj_dim = 256
        # )

        #id 9 is reserved for background and id "0" is not used, total 10
        # self.class_embed = nn.Linear(256, self.num_classes + 1)
        self.class_embed = MLP(transformer_dim, iou_head_hidden_dim, self.num_classes + 1, 3)
        self.bbox_embed = MLP(transformer_dim, iou_head_hidden_dim, 4, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hfc_embed:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        hs, _ = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            hfc_embed=hfc_embed,
        )
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out
        # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        # return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hfc_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forwarself.output_upscalingd' for more details."""
        output_tokens = self.mask_tokens.weight
        # output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        tokens = torch.repeat_interleave(output_tokens.unsqueeze(0), image_embeddings.shape[0], dim=0)

        # cross attend the dense embeddings with high frequency embeddings
        # dense_prompt_embeddings = self.hfc_attn(hfc_embed, dense_prompt_embeddings.permute(0,2,3,1)) 
        # Expand per-image data in batch direction to be per-mask
        #src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = image_embeddings
        # src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        # import pdb; pdb.set_trace()
        hs, _ = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        #mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        mask_tokens_out = hs[:, :(self.num_mask_tokens), :]

        return mask_tokens_out, iou_token_out


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class CrossAttentionHfcPrompt(nn.Module):
    """
    attend patch embeddings with high frequency componenets
    """
    def __init__(
            self,
            d_model = 1024,
            nhead = 8,
            dropout = 0.1,
            dim_feedforward = 1024,
            activation = 'relu',
            proj_dim = 256
    ):
        super().__init__()
        self.activation = F.relu
        self.proj_hfc = nn.Conv2d(d_model*4, proj_dim, (1,1))
        self.proj_patch = nn.Conv2d(d_model, proj_dim, (1,1))
        self.cross_attn = nn.MultiheadAttention(proj_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(proj_dim, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_feedforward)
        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(dim_feedforward)

        #position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, proj_dim, 64, 64))
    
    def forward(
            self, 
            hfc_embed, 
            patch_embed, 
    ):
        b, h, w, c = hfc_embed.shape
        c = 256 #output channels
        hfc_embed = self.proj_hfc(hfc_embed.permute(0,3,1,2)) + self.pos_embed
        patch_embed = self.proj_patch(patch_embed.permute(0,3,1,2))
        #flatten NxCxHxW to HWxNxC
        hfc_embed = hfc_embed.flatten(2).permute(2,0,1)
        patch_embed = patch_embed.flatten(2).permute(2,0,1)

        src2 = self.cross_attn(
            query = patch_embed, 
            key = hfc_embed,
            value = hfc_embed)[0]
        patch_embed = patch_embed + self.dropout1(src2)
        patch_embed = self.norm1(patch_embed)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(patch_embed))))
        # src2 = self.activation(self.linear1(patch_embed))
        src2 = src2 + self.dropout3(patch_embed)
        patch_embed = self.norm2(src2)

        patch_embed = patch_embed.permute(1, 2, 0).view(b, c, h, w)

        return patch_embed