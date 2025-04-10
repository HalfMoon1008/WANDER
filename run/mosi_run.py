import random
import torch
import argparse
from train import mosi_train
from dataloader.getloader import getdataloader
import numpy as np

# ---------------------------------------------------------------------
# 실험 실행 진입점
# ---------------------------------------------------------------------

def mosirun():
    parser = argparse.ArgumentParser(description="Wander")

    # ---------------------------------------------------------------
    # 실험에 필요한 파라미터 정의
    # ---------------------------------------------------------------

    # 사전학습 모델 (Wander 구조 위에 불러올)
    parser.add_argument("--pretrained_model", type=str, default="", help="path to pre-trained model")

    # 데이터셋 관련
    parser.add_argument("--dataset", type=str, default="mosi", help="dataset name")
    parser.add_argument("--data_path", type=str, default="mosi", help="dataset root path")

    # Dropout 설정
    parser.add_argument("--attn_dropout", type=float, default=0.2)
    parser.add_argument("--relu_dropout", type=float, default=0.2)
    parser.add_argument("--embed_dropout", type=float, default=0.15)
    parser.add_argument("--res_dropout", type=float, default=0.2)
    parser.add_argument("--out_dropout", type=float, default=0.1)

    # 모델 구조 (Transformer 관련)
    parser.add_argument("--nlevels", type=int, default=32, help="number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("--proj_dim", type=int, default=40, help="projected feature dimension")
    parser.add_argument("--attn_mask", action="store_false", help="disable attention masking")

    # Adapter 관련 하이퍼파라미터
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--drank", type=int, default=8)
    parser.add_argument("--trank", type=int, default=12)

    # 학습 설정
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--clip", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optim", type=str, default="AdamW")
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--when", type=int, default=10)

    # 로깅 및 디바이스 설정
    parser.add_argument("--log_interval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--no_cuda", action="store_true")

    # -----------------------------------------------------------------
    # 파라미터 파싱
    # -----------------------------------------------------------------

    args = parser.parse_args()
    dataset = str.lower(args.dataset.strip())

    # 시드 고정 함수 (재현성)
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # 디바이스 설정
    torch.set_default_tensor_type("torch.FloatTensor")
    if torch.cuda.is_available():
        if args.no_cuda:
            print("WARNING: You have CUDA, but disabled it with --no_cuda")
        else:
            use_cuda = True
    else:
        print("CUDA not available!")

    setup_seed(args.seed)

    # -----------------------------------------------------------------
    # 데이터 로딩 및 사전 정보 준비
    # -----------------------------------------------------------------

    dataloder, orig_dim, t_dim = getdataloader(args.dataset, args.data_path, args.batch_size)
    train_loader = dataloder["train"]
    valid_loader = dataloder["valid"]
    test_loader = dataloder["test"]

    # 하이퍼파라미터 세팅 객체
    hyp_params = args
    hyp_params.orig_dim = orig_dim          # 각 modality 입력 차원
    hyp_params.t_dim = t_dim                # 각 modality 시퀀스 길이
    hyp_params.layers = args.nlevels        # Transformer 레이어 수
    hyp_params.use_cuda = use_cuda
    hyp_params.dataset = dataset
    hyp_params.when = args.when
    hyp_params.n_train = len(train_loader)
    hyp_params.n_valid = len(valid_loader)
    hyp_params.n_test = len(test_loader)
    hyp_params.output_dim = 1               # 감정 점수 예측 → regression
    hyp_params.criterion = "L1Loss"         # 감정 예측용 MAE (mean absolute error)
    hyp_params.num_mod = 3                  # modality 수: text + audio + vision

    # 학습 시작
    mosi_train.initiate(hyp_params, train_loader, valid_loader, test_loader)


# ---------------------------------------------------------------------
# 실행 시작점
# ---------------------------------------------------------------------
if __name__ == "__main__":
    mosirun()
