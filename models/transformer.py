import math
import torch
from torch import nn
import torch.nn.functional as F
from models.position_embedding import SinusoidalPositionalEmbedding
from models.multihead_attention import MultiheadAttention

# ------------------------------------------------------------------------
# TransformerEncoder: 여러 층으로 쌓인 인코더 블록
# ------------------------------------------------------------------------

class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()

        self.dropout = embed_dropout          # 임베딩 드롭아웃
        self.attn_dropout = attn_dropout      # 어텐션 스코어 드롭아웃
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)  # 스케일링 (논문에선 sqrt(d_model))
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)  # 위치 임베딩
        self.attn_mask = attn_mask            # future mask 여부 (default: False)

        # 여러 개의 인코더 레이어를 리스트로 쌓음
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=attn_mask
            )
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))  # 버전 정보 (사용 X)

        # 마지막 layer norm 적용 여부
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in):
        """
        x_in: (B, T, D) - 모델 입력 (token + 임베딩 상태)
        반환값: (B, T, D) - Transformer 인코더를 통과한 출력
        """

        # 1. 임베딩 스케일링
        x = self.embed_scale * x_in

        # 2. 포지셔널 임베딩 추가
        if self.embed_positions is not None:
            # → positional embedding은 (B, T, D) → transpose하여 (T, B, D)로 정렬
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)

        # 3. 드롭아웃 적용
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 4. 각 레이어 통과
        intermediates = [x]
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        # 5. 마지막 layer normalization
        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """지원 가능한 최대 길이 반환"""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

# ------------------------------------------------------------------------
# TransformerEncoderLayer: 하나의 인코더 층 (Self-Attn + FFN)
# ------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. Self-attention 모듈 (Multi-head)
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.attn_mask = attn_mask

        # 2. Feedforward 파트용 드롭아웃
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True  # Pre-norm 구조 사용

        # 3. Feedforward layer (2-layer MLP)
        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)     # 확장
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)     # 다시 축소

        # 4. LayerNorm 2개: self-attn & ffn 각각
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x):
        # ---------------------------
        # 1. Self-Attention Block
        # ---------------------------
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)

        # self-attn: Q = K = V = x
        x, _ = self.self_attn(query=x, key=x, value=x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        # ---------------------------
        # 2. Feedforward Block
        # ---------------------------
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)

        x = F.relu(self.fc1(x))  # ReLU + 확장
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)          # 축소
        x = F.dropout(x, p=self.res_dropout, training=self.training)

        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        """
        Pre-LN 또는 Post-LN 구조 선택 지원
        → normalize_before가 True이면 Pre-Norm
        """
        assert before ^ after  # 둘 중 하나만 True여야 함
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

# ------------------------------------------------------------------------
# 기타 유틸 함수
# ------------------------------------------------------------------------

def fill_with_neg_inf(t):
    """float('-inf')로 텐서를 채우는 함수 (FP16 호환성 보장)"""
    return t.float().fill_(float('-inf')).type_as(t)

def buffered_future_mask(tensor, tensor2=None):
    """
    future token을 마스킹하기 위한 upper-triangular matrix 생성
    causal masking (예: decoder에서 사용됨)
    """
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]

def Linear(in_features, out_features, bias=True):
    """선형 계층 초기화 함수 (xavier + bias 0)"""
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def LayerNorm(embedding_dim):
    """LayerNorm 생성자 → 별도 함수로 분리해서 깔끔하게 사용"""
    return nn.LayerNorm(embedding_dim)
