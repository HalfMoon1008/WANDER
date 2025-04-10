import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    """Multi-headed attention. 
    "Attention Is All You Need" 논문 기반 구조로, 
    query, key, value 간 내적을 통해 context vector를 생성하는 구조.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim                # 전체 embedding 차원
        self.num_heads = num_heads                # head 수
        self.attn_dropout = attn_dropout          # attention weight dropout
        self.head_dim = embed_dim // num_heads    # head당 차원
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5      # dot-product scale factor

        # W_q, W_k, W_v를 하나로 묶은 projection weight (3 * embed_dim, embed_dim)
        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))

        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))  # bias도 projection에 같이 사용

        # 출력 projection layer (W_o)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # optional: key/value에 bias를 추가하는 경우
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn  # zero-padding key/value 추가 여부

        self.reset_parameters()  # 가중치 초기화

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """
        query, key, value: (T, B, D) 형식
        attn_mask: 마스킹 (선택적)
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()  # self-attention 여부
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None  # (사용되지 않음)

        # 1. Q, K, V 생성 (self, cross 등 상황에 따라 분기)
        if qkv_same:
            q, k, v = self.in_proj_qkv(query)  # W_qkv 통합 projection
        elif kv_same:
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        # 2. Q scaling (dot-product attention의 안정성)
        q = q * self.scaling

        # 3. key/value bias가 있을 경우 concat
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )

        # 4. head 수만큼 쪼개고 transpose → shape: (B * H, T, d_k)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        # 5. zero attention 옵션이 있으면 key/value 뒤에 0 추가
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )

        # 6. attention score 계산 (QK^T) → shape: (B*H, T_q, T_k)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # 7. attention mask 적용 (선택)
        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        # 8. softmax → dropout
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        # 9. weighted sum (Attention(Q,K,V) = softmax(QK^T)V)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # 10. heads 합치고 output projection
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # 11. attention weight: head 평균 (시각화 등 사용 가능)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def in_proj_qkv(self, query):
        """
        query에서 Q, K, V를 한꺼번에 projection하는 함수
        - self-attention에서 사용됨
        - in_proj_weight: [3*D, D] → W_q, W_k, W_v가 하나로 합쳐져 있음
        - chunk(3, dim=-1): Q/K/V를 마지막 차원에서 3등분
        """
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        """
        key에서 K, V만 projection하는 함수
        - encoder-decoder attention에서 사용
        - W_k, W_v는 in_proj_weight의 뒷부분에 있음
        - chunk(2): 마지막 차원 기준으로 2등분 → (K, V)
        """
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        """
        query에서 Q만 projection하는 함수
        - in_proj_weight의 앞부분 사용 (0 ~ embed_dim)
        - kwargs로 직접 weight, bias 지정 가능
        """
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        """
        key에서 K만 projection하는 함수
        - in_proj_weight의 중간 부분 사용 (embed_dim ~ 2*embed_dim)
        """
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        """
        value에서 V만 projection하는 함수
        - in_proj_weight의 마지막 부분 사용 (2*embed_dim ~)
        """
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        """
        projection 공통 함수 (Q, K, V에서 재사용됨)
        - start, end로 slicing 위치 지정
        - default: in_proj_weight / in_proj_bias 사용
        - 실제 projection은 F.linear로 수행
        """
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)

        # 필요한 부분만 잘라서 사용 (W_q, W_k, W_v)
        weight = weight[start:end, :]  # shape: (slice_dim, embed_dim)
        if bias is not None:
            bias = bias[start:end]     # shape: (slice_dim,)
        return F.linear(input, weight, bias)
