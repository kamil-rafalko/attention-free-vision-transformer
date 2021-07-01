import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.normalization import LayerNorm
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Shape(batch_size, in_channels, img_height, img_width)
        Returns:
            patch_embeddings: Shape(batch_size, n_patches, embedding_dim)
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class AttentionFreeTransformer(nn.Module):
    def __init__(self, dimension=768, num_heads=12):
        super(AttentionFreeTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(embed_dim=dimension)
        self.encoder = AttentionFreeTransformerEncoder(dimension)

    def forward(self, x):
        x = self.patch_embedding(x)
        return self.encoder(x)


class Attention(nn.Module):
    """
    Args:
        dim: int
            The input and output dimension of per token features
        n_heads: int
            Number of attention heads.
        qkv_bias: bool
            If True then we include bias to the query key and value projectsions
        attn_p: float
            Dropout probability applied to the query, key and value tensors.
        proj_p: float
            Dropout probability applied to the output tensor
    """
    def __init__(self, dim, n_heads=12, window_size=16, qkv_bias=True, attn_p=0., proj_p=0., eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, n_heads)
        self.to_v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

        self.wbias = nn.Parameter(torch.Tensor(n_heads, n_heads, window_size, window_size))  # TODO: check dimensions (n_patches + 1) - should be s ?
        nn.init.xavier_uniform_(self.wbias)
        self.eps = eps

        self.gamma = nn.Parameter(torch.Tensor(n_heads, n_heads, window_size, window_size))  # TODO choose some size
        self.beta = nn.Parameter(torch.Tensor(n_heads, n_heads, window_size, window_size))
        nn.init.zeros_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor
                Shape(n_samples, n_patches + 1, dim)
        Returns:
            torch.Tensor
                Shape(n_samples, n_patches + 1, dim)
        """
        # print("x has nan", torch.isnan(x).any())
        B, T, dim = x.shape

        q = self.to_q(x).view(B, T, self.n_heads, self.head_dim).permute((0, 2, 1, 3))
        k = self.to_k(x).view(B, T, self.n_heads).permute((0, 2, 1))
        v = self.to_v(x).view(B, T, self.n_heads, self.head_dim).permute((0, 2, 1, 3))

        self.wbias = nn.Parameter(self.gamma * (self.wbias - torch.mean(self.wbias)) / (torch.std(self.wbias) + self.eps) + self.beta)

        # qkv = self.qkv(x)  # (n_samples, n_patches + 1, dim * 3)
        # qkv = qkv.reshape(
        #     n_samples, n_tokens, 3, self.n_heads, self.head_dim
        # )  # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        # qkv = qkv.permute(
        #     2, 0, 3, 1, 4
        # )  # (3, n_samples, n_heads, n_patches + 1, head_dim)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        # temp_wbias = self.wbias[:n_tokens, :n_tokens].unsqueeze(0)

        # Atention
        q_sig = torch.sigmoid(q) # (n_samples, n_heads, n_patches + 1, head_dim)

        k_t = k.transpose(-2, -1)
        input = torch.mul(k.unsqueeze(3), v)
        input_pad = F.pad(input, (7, 8, 7, 8))
        dividend = torch.conv2d(input_pad, torch.exp(self.wbias) - 1) + torch.sum(torch.mul(torch.exp(k).unsqueeze(3), v), dim=2, keepdim=True)
        k_pad = F.pad(k.unsqueeze(3), (7, 8, 7, 8))
        devisior = torch.conv2d(k_pad, torch.exp(self.wbias) - 1) + torch.sum(k.unsqueeze(3), dim=2, keepdim=True)

        weighted = dividend / (devisior + self.eps)

        # print("q_sig has nan", torch.isnan(q_sig).any())
        # print("weighted has nan", torch.isnan(weighted).any())

        Yt = torch.mul(q_sig, weighted)
        Yt = Yt.transpose(
            1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        Yt = Yt.flatten(2) # (n_samples, n_patches + 1, dim)

        Yt = self.proj(Yt)
        # Yt = self.proj_drop(Yt)

        # print("Yt has nan:", torch.isnan(Yt).any())

        # print("Yt:", Yt)
        return Yt


        # k_t = k.transpose(-2, -1)
        # dp = (
        #     q @ k_t
        # ) * self.scale  # dot product (n_samples, n_heads, n_patches + 1, n_patches + 1)
        # attn = dp.softmax(dim=-1) # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        # attn = self.attn_drop(attn)
        #
        # weighted_avg = attn @ v
        # weighted_avg = weighted_avg.transpose(
        #     1, 2
        # )  # (n_samples, n_patches + 1, n_heads, head_dim)
        # weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches + 1, dim)
        #
        # x = self.proj(weighted_avg)
        # x = self.proj_drop(x)
        #
        # return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        # print("x mlp", torch.isnan(x).any())
        x = self.fc1(
            x
        ) # (n_samples, n_patches + 1, hidden_features)
        # print("x mlp after fc1", torch.isnan(x).any())
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        # print("x mlp after act", torch.isnan(x).any())
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        # print("x mlp after drop", torch.isnan(x).any())
        x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
        # print("x mlp after fc2", torch.isnan(x).any())
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        # print("x mlp after drop2", torch.isnan(x).any())
        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, window_size=16, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )

    def forward(self, x):
        # print("x block has nan", torch.isnan(x).any())
        x_norm_1 = self.norm1(x)
        x = x + self.attn(x_norm_1)
        # print("x block after attn has nan", torch.isnan(x).any())
        # print("Norm weight:", self.norm2.weight, "norm bias:", self.norm2.bias)
        x_norm = self.norm2(x)
        # print("x_norm block after attn has nan", torch.isnan(x_norm).any())
        x = x + self.mlp(x_norm)
        # print("x block after mlp has nan", torch.isnan(x).any())

        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=384,
            patch_size=16,
            in_chans=3,
            n_classes=1000,
            embed_dim=768,
            depth=12,
            n_heads=12,
            window_size=16,
            mlp_ratio=4,
            qkv_bias=True,
            p=0.,
            attn_p=0.
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        # cls_token = self.cls_token.expand(
        #     n_samples, -1, -1
        # )
        # x = torch.cat((cls_token, x), dim=1)
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        # print("x transformer has nan", torch.isnan(x).any())
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)

        return x


class AttentionFreeTransformerEncoder(nn.Module):
    def __init__(self, dimension: int):
        super(AttentionFreeTransformerEncoder, self).__init__()
        self.dimension = dimension
        self.qkv = nn.Linear(dimension, dimension * 3, bias=False)
        self.transformer_conv = AttentionFreeTransformerConv(dimension)
        self.norm = LayerNorm(dimension)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.dimension).permute(2, 0, 3, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x2 = self.transformer_conv(q, k, v)
        x2 = x + self.norm(x2)
        return x2


class AttentionFreeTransformerConv(nn.Module):
    def __init__(self, dimension: int, window_size=16):
        super(AttentionFreeTransformerConv, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(dimension, dimension, window_size))

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return torch.sigmoid(query) * \
               (torch.conv2d(key * value, torch.exp(self.weights) - 1, stride=1) + torch.sum(torch.exp(key) * value)) / \
               (torch.conv2d(key, torch.exp(self.weights) - 1, stride=1) + torch.sum(torch.exp(key)))


