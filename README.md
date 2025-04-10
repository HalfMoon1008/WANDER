# Wander: 멀티모달 태스크를 위한 경량 어댑터

- **Food-101 분류** (이미지 + 텍스트)
- **CMU-MOSI 감정 분석 회귀** (텍스트 + 오디오 + 비디오)

---

## 프로젝트 구조

```
Wander/
├── dataloader/                  # 데이터셋 로딩 관련 코드
│   ├── fooddata.py              # Food-101용 데이터 클래스 (이미지 + 텍스트)
│   ├── mosidata.py              # MOSI/MOSEI용 데이터 클래스 (텍스트 + 오디오 + 비디오)
│   └── getloader.py             # 데이터셋별 공통 로딩 인터페이스 (train/valid/test 반환)
│
├── models/                      # 전체 모델 구조 정의
│   ├── adapter.py               # Wander 어댑터의 핵심 구현 (low-rank fusion 구조)
│   ├── foodmodel.py             # Food-101 태스크용 vision + text 기반 모델
│   ├── model.py                 # MOSI 실험용 AdapterModel 및 LateFusion 모델 정의
│   ├── multihead_attention.py   # Multi-head Attention 직접 구현
│   ├── position_embedding.py    # 위치 임베딩 (Sinusoidal 방식)
│   └── transformer.py           # Transformer 인코더 레이어 정의
│
├── train/                       # 학습 루프 정의
│   ├── food_train.py            # Food-101 학습 및 평가 루프
│   └── mosi_train.py            # MOSI 감정 분석 학습 루프
│
├── run/                         # 실험 실행 스크립트
│   ├── food_run.py              # Food-101 학습 실행 (argparse 기반)
│   └── mosi_run.py              # MOSI 학습 실행 (argparse 기반)
│
├── utils/                       # 평가 및 공용 유틸 함수
│   ├── eval_metrics.py          # 평가 지표 함수 (accuracy, F1, MAE 등)
│   └── util.py                  # 파라미터 수 계산, 모델 weight 불러오기 등 유틸
```

---

Original source:https://github.com/zrguo/Wander
