import torch
import torch.nn as nn
import math

from models.ViT_helper import DropPath, to_2tuple, trunc_normal_
from models.diff_aug import DiffAugment

class matmul(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        x = x1@x2
        return x

def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])
    

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W


class NestedTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.head_dim = d_model // n_heads

        self.attention_type = 'cross_multi_scale'
        self.query_size = 1
        self.n_blocks = [(8, 8), (7, 9), (8, 7), (7, 8), (8, 9), (9, 7), (9, 8), (9, 9)]
        self.max_block = 64
        assert len(self.n_blocks) == self.n_heads
        
        self.linear0 = nn.Linear(3, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.query_embed = nn.Parameter(
            torch.randn(1, self.n_heads, self.max_block, self.query_size + 1, self.head_dim))
        self.o_z = nn.Parameter(torch.zeros([1, self.n_heads, self.max_block, 1, self.head_dim]))

        self.o_pos_q1 = nn.Parameter(torch.randn(1, self.n_heads, self.max_block, self.query_size, self.head_dim)) # b * 8 * 64 * 1 * 32
        self.o_pos_k1 = nn.Parameter(torch.randn(1, self.n_heads, self.max_block, self.query_size, self.head_dim)) # b * 8 * 64 * 1 * 32
        
        self.out = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def self_attn(self, x):
        x = self.linear0(x) # b * 32 * 32 * 256
        k = self.key(x) # b * 32 * 32 * 256
        qkv = self.qkv(x) # b * 32 * 32 * (256*3)
        x = torch.cat([k, qkv])  # b * 32 * 32 * (256*4)
        
        bs, h, w, c = x.shape  # b * 32 * 32 * (256*4)
        nh, nw = self.n_blocks[0]
        x = x.view(bs, nh, h // nh, nw, w // nw, c).transpose(2, 3) # b * 8 * 4 * 8 * 4 * (256*4)
        x = x.transpose(2, 3) # b * 8 * 8 * 4 * 4 * (256*4)
        x = x.reshape(bs, nh * nw, -1, c) # b * 64 * 16 * (256*4)
        x = x.view(bs, nh * nw, -1, self.n_heads, c/self.n_heads).permute(0, 3, 1, 2, 4)  # b * 8 * 64 * 16 * (32*4)
        x = x.reshape(bs, (self.n_heads * nh * nw), -1, c/self.n_heads) # b * (8*64) * 16 * (32*4)
        k, qkv = x.split([self.d_head, self.d_head * 3], dim=-1) # b * (8*64) * 16 * (32), b * (8*64) * 16 * (32*3)
        
        results = []
        q = self.query_embed.expand(bs, self.n_heads, self.max_block, self.query_size + 1, -1)  # b * 8 * 64 * 2 * 32
        q = q.reshape(bs, self.n_heads * self.max_block, self.query_size + 1, -1) # b * (8*64) * 2 * 32
        E = torch.eye(self.head_dim).to(device=q.device, dtype=q.dtype)
        
        a_logits = torch.matmul(q, k.transpose(2, 3)) # b * (8*64) * 2 * 16
        a_logits_row, _ = a_logits.split([self.query_size, 1], dim=2) # b * (8*64) * 1 * 16
        a_row = F.softmax(a_logits_row, dim=-1)  # b * (8*64) * 1 * 16
        a_col = F.softmax(a_logits.transpose(2, 3), dim=-1)  # b * (8*64) * 16 * 2
        q1, k1, v1 = torch.matmul(a_row, qkv).view(bs, self.n_heads, self.max_block, self.query_size, -1).split(self.head_dim, dim=-1) # b * 8 * 64 * 1 * 32
        q1 = q1 + self.o_pos_q1
        k1 = k1 + self.o_pos_k1
        q1 = q1.view(bs * self.n_heads, self.max_block * self.query_size, -1) # (b*8) * (64*1) * 32
        k1 = k1.view(bs * self.n_heads, self.max_block * self.query_size, -1) # (b*8) * (64*1) * 32
        v1 = v1.view(bs * self.n_heads, self.max_block * self.query_size, -1) # (b*8) * (64*1) * 32

        o = F.multi_head_attention_forward(
            q1.transpose(0, 1), k1.transpose(0, 1), v1.transpose(0, 1),
            self.head_dim, 1, None, None, None, None, False,
            self.dropout, E, None, use_separate_proj_weight=True,
            q_proj_weight=E, k_proj_weight=E, v_proj_weight=E,
            training=self.training)[0].transpose(0, 1) # (b*8) * (64*1) * 32

        o = o.reshape(bs, self.n_heads * self.max_block, self.query_size, -1) # b * (8*64) * 1 * 32
        o_z = self.o_z.expand(bs, self.n_heads, self.max_block, 1, -1)  # b * 8 * 64 * 1 * 32
        o_z = o_z.reshape(bs, self.n_heads * self.max_block, 1, -1) # b * (8*64) * 1 * 32
        o = torch.cat([o, o_z], dim=2) # b * (8*64) * 2 * 32
        o = torch.matmul(a_col, o) # b * (8*64) * 16 * 32

        res = o.reshape(bs, self.max_block, -1, self.n_heads*self.n_models) # b * 64 * 16 * 256
        res = self.out(res)
        return res

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, x):
        # self attention
        x = self.norm1(x + self.dropout1(self.self_attn(x)))

        # ffn
        x = self.forward_ffn(x)
        
        return src_reshaped


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Discriminator(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, args, img_size=32, patch_size=4, in_chans=3, num_classes=1, embed_dim=None, depth=7,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim = self.embed_dim = args.df_dim  # num_features for consistency with other models
        depth = args.d_depth
        self.args = args
#         patch_size = args.patch_size
#         if hybrid_backbone is not None:
#             self.patch_embed = HybridEmbed(
#                 hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
#         else:
#             self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
#         num_patches = (args.img_size // patch_size)**2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if "None" not in self.args.diff_aug:
            x = DiffAugment(x, self.args.diff_aug, True)
        B = x.shape[0]
#         x = self.patch_embed(x).flatten(2).permute(0,2,1)

#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:,0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


# def vit_small_patch16_224(pretrained=False, drop_rate=0., drop_path_rate=0., **kwargs):
#     if pretrained:
#         # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
#         kwargs.setdefault('qk_scale', 768 ** -0.5)
#     model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., drop_rate=drop_rate, drop_path_rate=drop_path_rate, **kwargs)
#     model.default_cfg = default_cfgs['vit_small_patch16_224']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
#     return model
