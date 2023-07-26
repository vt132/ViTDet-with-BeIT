import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig


from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12, 
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12, 
        checkpoint_activations=checkpoint_activations, 
    )


def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16, 
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24, 
        checkpoint_activations=checkpoint_activations, 
    )


class BEiT3Backbone(nn.Module):
    """BEiT 3 backbone for feature extraction."""
    def __init__(
            self,
            norm_layer=nn.LayerNorm,
            **kwargs):
        super(BEiT3Backbone, self).__init__()
        self.args = EncoderConfig(**kwargs)
        self.beit3 = BEiT3(self.args)
        self.apply(self._init_weights)
        embed_dim = self.args.encoder_embed_dim
        self.fc_norm = norm_layer(embed_dim)

        self.fc_norm.apply(self._init_weights)
        init_scale = 0.001
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pretrained = "beit3_base_patch16_224.pth"
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            print(f"load from {pretrained}")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, image, **kwargs):
        x = self.beit3(textual_tokens=None, visual_tokens=image)["encoder_out"]
        t = x[:, 1:, :]
        feature = self.fc_norm(t.mean(1))
        return feature

@BACKBONES.register_module()
class BEiT3BaseBackbone(BEiT3Backbone):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

