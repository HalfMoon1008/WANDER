import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, num_mod, dim, t_dim, rank, drank, trank) -> None:
        """
        Wander Adapter 클래스
        멀티모달 입력을 low-rank 방식으로 융합하기 위한 어댑터 구조

        Args:
            num_mod: modality 수 (예: 3 → text, audio, vision)
            dim: 각 modality의 feature 차원 (모두 동일하다고 가정)
            t_dim: 각 modality의 시퀀스 길이 리스트 (예: [seq_len_text, seq_len_audio, ...])
            rank: down projection 차원 수 (공통 bottleneck)
            drank: modality-specific 융합 rank (dim 기준)
            trank: modality-specific 융합 rank (time 기준)
        """
        super().__init__()
        self.n_mod = num_mod       # modality 개수
        self.rank = rank           # 공통 bottleneck 차원
        self.drank = drank         # feature 차원 압축 후 rank
        self.trank = trank         # 시계열 차원(rank)
        self.t_dim = t_dim         # 각 modality의 time length

        # 각 modality 별로 down projection layer 생성 (dim → rank)
        self.down = nn.ParameterList(
            [
                nn.Linear(dim, rank)
                for _ in range(self.n_mod)
            ]
        )

        # modality-specific low-rank factor (dim 기반)
        # shape: (drank, rank, dim)
        # → 각 modality에 대해 drank개의 low-rank 구성 요소
        self.factor_dim_list = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.drank, rank, dim))  # 초기화: 랜덤값
                for _ in range(self.n_mod)
            ]
        )

        # modality-specific low-rank factor (time 기반)
        # shape: (trank, time_length, rank)
        self.factor_t_list = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.trank, t_dim[i], rank))  # 초기화: 0으로
                for i in range(self.n_mod)
            ]
        )

    def forward(self, x):
        """
        입력: x는 modality별 입력 텐서의 리스트 (길이 = num_mod)
             각 텐서 shape: (B, T, D)  → batch, time, dim

        출력: 멀티모달 융합된 텐서 (shape: B, T, D)
        """

        # original: modality들을 concat해서 residual로 쓸 준비
        original = torch.cat(x, dim=1)  # shape: (B, T1+T2+..., D)

        # 1. 각 modality별로 down projection (dim → rank)
        for i in range(self.n_mod):
            x[i] = self.down[i](x[i])          # shape: (B, T, rank)
            x[i] = nn.functional.gelu(x[i])    # 비선형성 부여

        # 2. 각 modality별로 low-rank factor를 이용해 융합 특징 생성
        fusion_feat = []
        for i in range(self.n_mod):
            # feature 축 융합
            # factor_dim: (drank, rank, dim) → 합치면 (rank, dim)
            wd = self.factor_dim_list[i].sum(dim=0)  # shape: (rank, dim)

            # x[i]: (B, T, rank) @ (rank, dim) → (B, T, dim)
            tmp_feat = torch.matmul(x[i], wd)

            # 시계열 축 융합을 위해 (B, T, dim) → (B, dim, T)
            tmp_feat = tmp_feat.permute(0, 2, 1)

            # 시간 축 factor_t: (trank, T, rank) → 합치면 (T, rank)
            wt = self.factor_t_list[i].sum(dim=0)  # shape: (T, rank)

            # (B, dim, T) @ (T, rank) → (B, dim, rank)
            tmp_feat = torch.matmul(tmp_feat, wt)

            fusion_feat.append(tmp_feat)

        # 3. modality-wise element-wise fusion (곱 연산)
        fusion = fusion_feat[0]
        for i in range(1, self.n_mod):
            fusion *= fusion_feat[i]  # modality 간의 조합 효과 생성 (곱셈으로)

        # 4. 원래 시간 축으로 되돌리기 (B, dim, rank) → (B, rank, dim)
        fusion = fusion.permute(0, 2, 1)

        # 5. Residual 연결 (original 일부 추가)
        # original에서 필요한 시간 길이만큼 잘라서 residual로 더함
        fusion += original[:, : fusion.shape[1], :]  # (B, T, D)

        return fusion
