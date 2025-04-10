import torch
import torch.nn as nn
from transformers import ViTModel, BertModel
import torch.nn.functional as F
from models.adapter import Adapter  # Wander Adapter
from models.transformer import TransformerEncoder  # 내부 융합용 Transformer

class FoodModel(nn.Module):
    def __init__(self, vis_path='ViT', text_path='BERT',
                 output_dim=101, out_dropout=0.1, embed_dim=768, num_heads=8, layers=2):
        """
        이미지-텍스트 멀티모달 분류용 기본 모델

        Args:
            vis_path, text_path: 사전학습된 ViT, BERT 모델 경로
            output_dim: 분류 클래스 수 (Food101 → 101)
            out_dropout: 출력 계층 dropout
            embed_dim: Transformer 내부 embedding 크기
            num_heads, layers: Transformer 융합층 설정
        """
        super(FoodModel, self).__init__()
        self.num_mod = 2  # modality 수: 이미지 + 텍스트
        self.out_dropout = out_dropout

        # 사전학습된 모델 로딩
        self.vision_encoder = ViTModel.from_pretrained(vis_path)
        self.text_encoder = BertModel.from_pretrained(text_path)

        # 사전학습된 모델은 freeze
        self.vision_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # 이미지 + 텍스트 융합을 위한 Transformer
        self.fusion = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, layers=layers)

        # classification을 위한 linear layer
        self.proj1 = nn.Linear(768, 768)
        self.out_layer = nn.Linear(768, output_dim)

    def get_embed(self, v, t):
        """
        ViT와 BERT로부터 마지막 hidden state를 추출하고 concat한 뒤,
        Transformer로 융합 → modality별 feature 분리 반환
        """
        ti, ta, tt = t  # text input: input_ids, attention_mask, token_type_ids

        v = self.vision_encoder(v)["last_hidden_state"]  # shape: (B, T_v, 768)
        t = self.text_encoder(ti, ta, tt)["last_hidden_state"]  # shape: (B, T_t, 768)

        feature = torch.cat([v, t], dim=1)  # 융합 전 concat (B, T_v+T_t, 768)
        fusion = self.fusion(feature)       # Transformer 융합

        v_f = fusion[:, :v.shape[1], :]     # 이미지 부분
        t_f = fusion[:, v.shape[1]:, :]     # 텍스트 부분
        return v_f, t_f

    def forward(self, v, t):
        """
        전체 forward 흐름: 입력 → 융합 → proj → output

        입력:
            v: 이미지 tensor (B, 3, 224, 224)
            t: 텍스트 tuple (input_ids, attn_mask, token_type_ids)

        출력:
            output logits (B, 101)
        """
        ti, ta, tt = t
        v = self.vision_encoder(v)["last_hidden_state"]
        t = self.text_encoder(ti, ta, tt)["last_hidden_state"]

        feature = torch.cat([v, t], dim=1)
        fusion = self.fusion(feature)

        # CLS 토큰 기준으로 분류
        cls_h = fusion[:, 0, :]  # 첫 토큰 위치

        # proj1 + dropout → 최종 분류 계층
        last_hs_proj = F.dropout(
            F.gelu(self.proj1(cls_h)), p=self.out_dropout, training=self.training
        )
        output = self.out_layer(last_hs_proj)
        return output


class FoodModelWander(nn.Module):
    def __init__(
        self, vis_path='ViT', text_path='BERT',
        output_dim=101, t_dim=[], rank=8, drank=8, trank=8, out_dropout=0.1
    ):
        """
        Wander 어댑터 기반 FoodModel
        - Base는 FoodModel을 사용하되, middle 융합을 Adapter로 대체함

        Args:
            t_dim: 각 modality의 시퀀스 길이 (ViT, BERT)
            rank, drank, trank: 어댑터 관련 차원 설정
        """
        super(FoodModelWander, self).__init__()
        self.num_mod = 2
        self.out_dropout = out_dropout

        # 기본 베이스 모델 정의
        self.basemodel = FoodModel(
            vit_path=vis_path,
            bert_path=text_path,
            output_dim=output_dim,
            out_dropout=out_dropout,
        )

        # 사전학습된 base는 freeze (Adapter만 학습)
        self.basemodel.requires_grad_(False)

        # classification head만 학습 가능하도록 다시 활성화
        self.basemodel.proj1.requires_grad_(True)
        self.basemodel.out_layer.requires_grad_(True)

        # 핵심: Wander Adapter (멀티모달 융합 수행)
        self.adapter = Adapter(self.num_mod, 768, t_dim, rank, drank, trank)

    def forward(self, v, t):
        """
        forward 흐름:
        1. ViT + BERT → 각각 feature 추출
        2. Adapter를 통해 융합
        3. 분류 head로 output 생성
        """
        v, t = self.get_embed(v, t)              # get_embed는 basemodel에서 사용
        fusion = self.adapter([v, t])            # Wander Adapter로 융합

        # fusion 결과에서 첫 토큰 위치 추출 후 투영 + 분류
        last_hs_proj = F.dropout(
            F.relu(self.basemodel.proj1(fusion[:, 0, :])), p=self.out_dropout, training=self.training
        )
        output = self.basemodel.out_layer(last_hs_proj)
        return output

