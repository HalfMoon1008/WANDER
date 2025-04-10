import math
import torch
import torch.nn as nn

# Code adapted from the fairseq repo.
# → fairseq 스타일의 sinusoidal positional embedding 구현체

# ------------------------------------------------------------------------
# 1. make_positions 함수
# ------------------------------------------------------------------------

def make_positions(tensor, padding_idx, left_pad):
    """
    입력 tensor에서 padding이 아닌 위치에 대해 positional index를 만들어줌
    → padding 위치는 무시하고, 나머지는 [padding_idx + 1]부터 번호를 매김

    Args:
        tensor: 입력 텐서 (shape: B x T)
        padding_idx: 패딩 토큰 인덱스
        left_pad: padding이 왼쪽에 있는지 여부 (True면 left padding)

    Returns:
        positional_tensor: (B x T) → padding 제외 위치에만 position 인덱스가 들어감
    """
    max_pos = padding_idx + 1 + tensor.size(1)  # 최대 position 번호
    device = tensor.get_device()               # GPU 번호

    # 장치마다 캐시된 range tensor 사용
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())  # 텐서 새로 생성
    setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))

    # position 인덱스를 미리 생성해둠 (padding_idx+1 부터)
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))

    # padding이 아닌 위치 마스크 생성
    mask = tensor.ne(padding_idx)  # padding이 아닌 곳만 True

    # expand된 position 텐서: (B x T)
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)

    if left_pad:
        # 왼쪽 패딩이면 position 값을 오른쪽으로 밀어야 함
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)

    # 새 텐서를 만들고, 마스크된 위치에 position 값만 할당
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


# ------------------------------------------------------------------------
# 2. SinusoidalPositionalEmbedding 클래스
# ------------------------------------------------------------------------

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Positional Embedding을 sin/cos 기반으로 생성하는 모듈
    - Attention Is All You Need 논문 방식
    - padding index는 무시함
    - GPU device에 따라 임베딩을 캐시해두는 구조 (DataParallel 대응)
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()  # GPU별 임베딩 캐시 저장소

        # dummy 텐서 등록 (타입 캐스팅용)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """
        sinusoidal embedding 행렬을 생성함

        Args:
            num_embeddings: position 개수
            embedding_dim: 임베딩 차원 수
            padding_idx: 해당 위치는 0으로 설정

        Returns:
            Tensor of shape (num_embeddings, embedding_dim)
        """
        half_dim = embedding_dim // 2

        # 각 차원별 주파수 비율 계산
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)

        # position index * frequency → sin, cos 생성
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)  # (P x D/2)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)

        if embedding_dim % 2 == 1:
            # 홀수 차원인 경우, 마지막에 0을 붙여서 shape 맞춤
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)

        if padding_idx is not None:
            # padding index 위치는 0으로 설정
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input):
        """
        입력: (B x T) → token ID 또는 padding index로 구성된 입력
        출력: (B x T x D) → position embedding 반환

        - 현재 device에 캐시된 임베딩이 없거나 부족하면 새로 생성
        - make_positions로 각 토큰의 position index 계산
        - index_select로 각 위치에 해당하는 임베딩 벡터 추출
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()

        if device not in self.weights or max_pos > self.weights[device].size(0):
            # 필요하면 임베딩 확장 생성
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )

        # dtype 정렬 (float32 등)
        self.weights[device] = self.weights[device].type_as(self._float_tensor)

        # 실제 position index 계산
        positions = make_positions(input, self.padding_idx, self.left_pad)

        # embedding lookup: (B*T, D) → (B, T, D)
        return self.weights[device].index_select(0, positions.contiguous().view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """지원 가능한 최대 포지션 수 (아주 크게 잡음)"""
        return int(1e5)  # 100,000
