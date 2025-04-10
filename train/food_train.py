import torch
from torch import nn
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.foodmodel import FoodModelWander
from utils.eval_metrics import eval_food
from utils.util import transfer_model, get_parameter_number

# ---------------------------------------------------------------
# 모델 초기화 및 학습 전체 수행 (entry point)
# ---------------------------------------------------------------

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    """
    전체 학습 과정의 진입점
    - 모델 초기화
    - 사전학습 weight 로드
    - optimizer, scheduler, loss 세팅
    - train_model 호출
    """
    model = FoodModelWander(
        hyp_params.pretrained_vit,
        hyp_params.pretrained_text,
        hyp_params.output_dim,
        hyp_params.t_dim,
        hyp_params.rank,
        hyp_params.drank,
        hyp_params.trank,
        hyp_params.out_dropout
    )

    # 🔹 사전학습된 모델 weight 불러오기
    transfer_model(hyp_params.pretrained_model, model)

    # 🔹 전체 파라미터 수 출력
    print(get_parameter_number(model))

    if hyp_params.use_cuda:
        model = model.cuda()

    # 🔹 optimizer 생성 (ex. Adam, SGD 등 문자열로 받음)
    optimizer = getattr(optim, hyp_params.optim)(
        model.parameters(), lr=hyp_params.lr, weight_decay=4e-5
    )

    # 🔹 loss 함수 설정 (ex. CrossEntropyLoss 등)
    criterion = getattr(nn, hyp_params.criterion)()

    # 🔹 learning rate scheduler 설정
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=hyp_params.when, factor=0.5, verbose=True
    )

    settings = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
    }

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

# ---------------------------------------------------------------
# 전체 학습 루프
# ---------------------------------------------------------------

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]

    # 🔹 학습 루틴 정의
    def train(model, optimizer, criterion):
        model.train()
        for i_batch, batch in enumerate(train_loader):
            text, image, batch_Y = batch
            eval_attr = batch_Y.squeeze(-1)

            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    # 텍스트 BERT 입력 분리
                    ti, ta, tt = (
                        text["input_ids"].cuda(),
                        text["attention_mask"].cuda(),
                        text["token_type_ids"].cuda(),
                    )
                    image, eval_attr = image.cuda(), eval_attr.cuda()
                    eval_attr = eval_attr.long()

            batch_size = image.size(0)

            # 🔸 큰 배치면 DataParallel 사용
            net = nn.DataParallel(model) if batch_size > 10 else model

            # 🔸 모델 forward
            preds = net(image, [ti, ta, tt])  # shape: (B, 101)
            preds = preds.view(-1, 101)
            eval_attr = eval_attr.view(-1)

            # 🔸 손실 계산 및 역전파
            raw_loss = criterion(preds, eval_attr)
            raw_loss.backward()

            # 🔸 gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)

            optimizer.step()

    # 🔹 평가 루틴 정의 (검증 or 테스트)
    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, batch in enumerate(loader):
                text, image, batch_Y = batch
                eval_attr = batch_Y.squeeze(dim=-1)

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        ti, ta, tt = (
                            text["input_ids"].cuda(),
                            text["attention_mask"].cuda(),
                            text["token_type_ids"].cuda(),
                        )
                        image, eval_attr = image.cuda(), eval_attr.cuda()
                        eval_attr = eval_attr.long()

                preds = model(image, [ti, ta, tt])
                preds = preds.view(-1, 101)
                eval_attr = eval_attr.view(-1)
                total_loss += criterion(preds, eval_attr).item()

                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    # -----------------------------------------------------------
    # 🔹 본격적인 학습 반복
    # -----------------------------------------------------------

    best_acc = 0
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()

        train(model, optimizer, criterion)
        val_loss, r, t = evaluate(model, criterion, test=False)
        acc = eval_food(r, t)  # 🔸 평가 지표 (F1, acc 등 포함)

        end = time.time()
        duration = end - start

        # 🔹 스케줄러로 learning rate decay
        scheduler.step(val_loss)

        print("-" * 50)
        print(f"Epoch {epoch:2d} | Time {duration:.4f} sec | Valid Loss {val_loss:.4f}")
        print("-" * 50)

        if acc > best_acc:
            best_acc = acc

    print("Best accuracy of validation: {:.4f}".format(best_acc))
