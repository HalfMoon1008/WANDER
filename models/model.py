import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import TransformerEncoder
from models.adapter import Adapter

# ------------------------------------------------------------------------
# Latefusion: 각 modality를 Transformer로 인코딩 후 feature-level에서 융합
# ------------------------------------------------------------------------

class Latefusion(nn.Module):
    def __init__(
        self,
        orig_dim,        # 각 modality의 입력 차원 리스트 (ex. [text_dim, audio_dim, video_dim])
        output_dim=1,    # 출력 차원 (회귀면 1, 분류면 클래스 수)
        proj_dim=40,     # 각 modality의 feature projection 차원
        num_heads=5,
        layers=5,
        relu_dropout=0.1,
        embed_dropout=0.15,
        res_dropout=0.1,
        out_dropout=0.1,
        attn_dropout=0.2,
    ):
        super(Latefusion, self).__init__()

        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mod = len(orig_dim)  # modality 수 (예: 3개)
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout

        # 🔹 1. 각 modality별 1D Conv projection layer (dim 정렬용)
        self.proj = nn.ModuleList([
            nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
            for i in range(self.num_mod)
        ])

        # 🔹 2. 각 modality별 독립적인 TransformerEncoder
        self.encoders = nn.ModuleList([
            TransformerEncoder(
                embed_dim=proj_dim,
                num_heads=self.num_heads,
                layers=self.layers,
                attn_dropout=self.attn_dropout,
                res_dropout=self.res_dropout,
                relu_dropout=self.relu_dropout,
                embed_dropout=self.embed_dropout,
            )
            for _ in range(self.num_mod)
        ])

        # 🔹 3. 출력 (분류/회귀용) MLP 헤드
        self.out_layer_proj0 = nn.Linear(3 * self.proj_dim, self.proj_dim)
        self.out_layer_proj1 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer_proj2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer = nn.Linear(self.proj_dim, output_dim)

    def get_emb(self, x):
        """
        입력 modality들을 Transformer로 각각 인코딩하여 반환
        x: modality별 리스트, 각 텐서 (B, T, D_i)
        반환값: 리스트 [h1, h2, h3], 각 h_i는 (B, T, proj_dim)
        """
        hs = list()
        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)               # (B, D, T)
            x[i] = self.proj[i](x[i])                 # Conv1D → (B, proj_dim, T)
            x[i] = x[i].permute(2, 0, 1)              # (T, B, proj_dim)
            h_tmp = self.encoders[i](x[i]).permute(1, 0, 2)  # (B, T, proj_dim)
            hs.append(h_tmp)
        return hs

    def get_res(self, x):
        """
        classification/regression output을 생성하는 헤드 부분
        x: (B, 3 * proj_dim)
        """
        last_hs = F.relu(self.out_layer_proj0(x))  # 1차 투영
        last_hs_proj = self.out_layer_proj2(
            F.dropout(
                F.relu(self.out_layer_proj1(last_hs)),  # 2차 투영
                p=self.out_dropout,
                training=self.training,
            )
        )
        last_hs_proj += last_hs  # residual 연결
        output = self.out_layer(last_hs_proj)
        return output

    def forward(self, x):
        """
        전체 forward 흐름:
        - 각 modality → 인코딩
        - 인코딩된 결과 → 첫 타임스텝만 뽑아서 concat
        - MLP 통과 후 output 반환
        """
        hs = list()
        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)
            x[i] = self.proj[i](x[i])
            x[i] = x[i].permute(2, 0, 1)
            h_tmp = self.encoders[i](x[i])
            hs.append(h_tmp[0])  # 첫 timestep만 사용 (CLS처럼)

        last_hs_out = torch.cat(hs, dim=-1)  # (B, 3 * proj_dim)
        last_hs = F.relu(self.out_layer_proj0(last_hs_out))
        last_hs_proj = self.out_layer_proj2(
            F.dropout(
                F.relu(self.out_layer_proj1(last_hs)),
                p=self.out_dropout,
                training=self.training,
            )
        )
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output

# ------------------------------------------------------------------------
# AdapterModel: Latefusion + Wander Adapter
# ------------------------------------------------------------------------

class AdapterModel(nn.Module):
    def __init__(
        self,
        orig_dim,         # 각 modality 입력 차원
        t_dim,            # 각 modality의 시퀀스 길이 (time_dim)
        rank, drank, trank,  # adapter 관련 차원 설정
        output_dim=1,
        proj_dim=40,
        num_heads=5,
        layers=5,
        relu_dropout=0.1,
        embed_dropout=0.15,
        res_dropout=0.1,
        out_dropout=0.1,
        attn_dropout=0.2,
    ):
        super().__init__()
        self.num_mod = len(orig_dim)

        # 🔹 base 모델: Latefusion 구조 (Transformer + MLP)
        self.basemodel = Latefusion(
            orig_dim,
            output_dim,
            proj_dim,
            num_heads,
            layers,
            relu_dropout,
            embed_dropout,
            res_dropout,
            out_dropout,
            attn_dropout,
        )

        # 🔹 Wander Adapter 추가 (멀티모달 융합 담당)
        self.adapter = Adapter(self.num_mod, proj_dim, t_dim, rank, drank, trank)

        # 🔹 base model freeze
        self.basemodel.requires_grad_(False)

        # 🔹 prediction head만 학습 가능하도록 설정
        self.basemodel.out_layer_proj0.requires_grad_(True)
        self.basemodel.out_layer_proj1.requires_grad_(True)
        self.basemodel.out_layer_proj2.requires_grad_(True)
        self.basemodel.out_layer.requires_grad_(True)

    def forward(self, x):
        """
        1. 각 modality를 Transformer로 인코딩
        2. Adapter로 멀티모달 융합
        3. 융합된 feature로 최종 prediction 수행
        """
        hs = self.basemodel.get_emb(x)          # → 각 modality별 인코딩 결과 리스트
        fusion = self.adapter(hs)               # → Wander adapter 융합 (B, T, D)

        # Adapter 출력: (B, D, T)로 변경 후, 각 modality의 [T=0] 값만 모음
        fusion = fusion.permute(0, 2, 1)        # (B, D, T)
        fusion = torch.cat([fusion[:, :, 0] for _ in range(self.num_mod)], dim=-1)  # (B, 3*D)

        output = self.basemodel.get_res(fusion)
        return output
