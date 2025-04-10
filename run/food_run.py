import random
import torch
import argparse
from train import food_train
import numpy as np
from dataloader.getloader import getdataloader

# ------------------------------------------------------------------------
# 실험 실행 함수 (entry point)
# ------------------------------------------------------------------------

def foodrun():
    parser = argparse.ArgumentParser(description="Wander")

    # 노트북 환경 대비용 파라미터 (jupyter 사용 시 필요)
    parser.add_argument("-f", default="", type=str)

    # -------------------------------------------------------
    # Task 관련 하이퍼파라미터
    # -------------------------------------------------------

    parser.add_argument("--pretrained_model", type=str, default="", help="path of the model to use")
    parser.add_argument("--pretrained_vit", type=str, default="", help="path of pre-trained vision encoder")
    parser.add_argument("--pretrained_text", type=str, default="", help="path of pre-trained text encoder")

    parser.add_argument("--dataset", type=str, default="food", help="dataset to use")
    parser.add_argument("--data_path", type=str, default="mosi", help="dataset path")

    # Adapter 관련 하이퍼파라미터
    parser.add_argument("--out_dropout", type=float, default=0.0, help="output layer dropout")
    parser.add_argument("--rank", type=int, default=64, help="adapter rank")
    parser.add_argument("--drank", type=int, default=8)
    parser.add_argument("--trank", type=int, default=12)

    # -------------------------------------------------------
    # 학습 설정
    # -------------------------------------------------------

    parser.add_argument("--batch_size", type=int, default=128, metavar="N", help="batch size")
    parser.add_argument("--clip", type=float, default=1.0, help="gradient clip value")
    parser.add_argument("--lr", type=float, default=2e-3, help="initial learning rate")
    parser.add_argument("--optim", type=str, default="AdamW", help="optimizer to use")
    parser.add_argument("--num_epochs", type=int, default=140, help="number of epochs")
    parser.add_argument("--when", type=int, default=30, help="when to decay learning rate")

    # -------------------------------------------------------
    # 로깅 및 시드 관련
    # -------------------------------------------------------

    parser.add_argument("--log_interval", type=int, default=30, help="log printing frequency")
    parser.add_argument("--seed", type=int, default=666, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="disable CUDA")

    args = parser.parse_args()

    # --------------------------------------------------------------------
    # 전처리
    # --------------------------------------------------------------------

    dataset = str.lower(args.dataset.strip())

    # 데이터셋에 따른 출력 차원 및 손실 함수 매핑
    output_dim_dict = {
        "food": 101,
    }
    criterion_dict = {
        "food": "CrossEntropyLoss"
    }

    # 시드 고정 함수 정의
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    torch.set_default_tensor_type("torch.FloatTensor")
    if torch.cuda.is_available():
        if args.no_cuda:
            print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
        else:
            use_cuda = True
    else:
        print("cuda not available!")

    setup_seed(args.seed)

    print(f"batch size is {args.batch_size}, epoch is {args.num_epochs}!!!!")

    # --------------------------------------------------------------------
    # 데이터 로딩 (getloader.py → DataLoader 딕셔너리 반환)
    # --------------------------------------------------------------------

    dataloder, orig_dim, t_dim = getdataloader(args.dataset, args.data_path, args.batch_size)
    train_loader = dataloder["train"]
    valid_loader = dataloder["valid"]
    test_loader = dataloder["test"]

    # --------------------------------------------------------------------
    # 하이퍼파라미터 객체 설정
    # --------------------------------------------------------------------

    hyp_params = args
    hyp_params.orig_dim = orig_dim             # [vision_dim, text_dim]
    hyp_params.use_cuda = use_cuda             # bool
    hyp_params.dataset = dataset
    hyp_params.when = args.when
    hyp_params.n_train = len(train_loader)
    hyp_params.n_valid = len(valid_loader)
    hyp_params.n_test = len(test_loader)
    hyp_params.output_dim = output_dim_dict.get(dataset, 1)
    hyp_params.criterion = criterion_dict.get(dataset, "L1Loss")
    hyp_params.num_mod = 2                     # modality 수 (이미지 + 텍스트)

    # --------------------------------------------------------------------
    # 학습 시작 (food_train.initiate() → train_model() 내부로 진입)
    # --------------------------------------------------------------------
    food_train.initiate(hyp_params, train_loader, valid_loader, test_loader)

# ------------------------------------------------------------------------
# 실행 시작
# ------------------------------------------------------------------------

if __name__ == "__main__":
    foodrun()
