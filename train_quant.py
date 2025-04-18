### 필요한 패키지 import
# 기본 유틸 및 딥러닝 관련 라이브러리 불러옴
import argparse
import logging
import os
import random
import sys
import time 
import copy

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn

# 사용자 정의 모듈 import
from models.custom_modules import *
from models.custom_models_resnet import *
from models.custom_models_vgg import *

# 유틸 함수들
from utils import *
import utils
from utils import printRed

# CIFAR 데이터셋 관련 함수
from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample

import utils_distill

# 시간 측정 시작
start_time = time.time()


### 문자열 → boolean으로 변환하는 함수 정의
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


### ArgumentParser를 통해 하이퍼파라미터 정의
parser = argparse.ArgumentParser(description="PyTorch Implementation of EWGS (CIFAR)")

# ---------- dataset 관련 ----------
parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10','cifar100'))
parser.add_argument('--arch', type=str, default='resnet20_quant')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--num_classes', type=int, default=10)

# ---------- 학습 설정 ----------
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--optimizer_m', type=str, default='Adam', choices=('SGD','Adam'))
parser.add_argument('--optimizer_q', type=str, default='Adam', choices=('SGD','Adam'))
parser.add_argument('--lr_m', type=float, default=1e-3)
parser.add_argument('--lr_q', type=float, default=1e-5)
parser.add_argument('--lr_m_end', type=float, default=0.0)
parser.add_argument('--lr_q_end', type=float, default=0.0)
parser.add_argument('--decay_schedule_m', type=str, default='150-300')
parser.add_argument('--decay_schedule_q', type=str, default='150-300')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--lr_scheduler_m', type=str, default='cosine', choices=('step','cosine'))
parser.add_argument('--lr_scheduler_q', type=str, default='cosine', choices=('step','cosine'))
parser.add_argument('--gamma', type=float, default=0.1)

# ---------- 양자화 관련 ----------
parser.add_argument('--QWeightFlag', type=str2bool, default=True)
parser.add_argument('--QActFlag', type=str2bool, default=True)
parser.add_argument('--weight_levels', type=int, default=2)
parser.add_argument('--act_levels', type=int, default=2)
parser.add_argument('--baseline', type=str2bool, default=False)
parser.add_argument('--bkwd_scaling_factorW', type=float, default=0.0)
parser.add_argument('--bkwd_scaling_factorA', type=float, default=0.0)
parser.add_argument('--use_hessian', type=str2bool, default=True)
parser.add_argument('--update_every', type=int, default=10)
parser.add_argument('--quan_method', type=str, default='EWGS')

# ---------- 기타 ----------
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--log_dir', type=str, default='./results/ResNet20_CIFAR10/W1A1/')
parser.add_argument('--load_pretrain', type=str2bool, default=False)
parser.add_argument('--pretrain_path', type=str, default='./results/ResNet20_CIFAR10/fp/checkpoint/last_checkpoint.pth')

# ---------- KD 관련 ----------
parser.add_argument('--distill', type=str, default=None, choices=['kd', 'crdst','hint', 'attention', 'similarity',
                                                    'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                    'rkd', 'pkt', 'abound', 'factor', 'nst'])
parser.add_argument('--teacher_path', type=str)
parser.add_argument('--teacher_arch', type=str)
parser.add_argument('--kd_T', type=float, default=4)
parser.add_argument('--kd_gamma', type=float, default=None)
parser.add_argument('--kd_alpha', type=float, default=None)
parser.add_argument('--kd_beta', type=float, default=None)
parser.add_argument('--kd_theta', type=float, default=None)

# ---------- NCE KD 관련 ----------
parser.add_argument('--feat_dim', default=128, type=int)
parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
parser.add_argument('--nce_k', default=16384, type=int)
parser.add_argument('--nce_t', default=0.1, type=float)
parser.add_argument('--nce_m', default=0.5, type=float)
parser.add_argument('--head', default='linear', type=str, choices=['linear', 'mlp', 'pad'])

# ---------- hint layer ----------
parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

# ---------- crdst (CKTF) ----------
parser.add_argument('--st_method', type=str, default='Last', choices=['Last', 'Smallest', 'Largest', 'First', 'Random'])

# ---------- 두 단계 방식 ----------
parser.add_argument('--init_epochs', type=int, default=30)

# 파라미터 파싱
args = parser.parse_args()
arg_dict = vars(args)

# ---------- 로그 디렉토리 및 로그 파일 생성 ----------
if not os.path.exists(args.log_dir):
    os.makedirs(os.path.join(args.log_dir, 'checkpoint'))

logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
                    level=logging.INFO,
                    format='')

log_string = 'configs\n'
for k, v in arg_dict.items():
    log_string += "{}: {}\t".format(k,v)
    print("{}: {}".format(k,v), end='\t')
logging.info(log_string+'\n')
print('')


# ---------- GPU 세팅 ----------
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------- seed 고정 ----------
if args.seed is not None:
    print("The seed number is set to", args.seed)
    logging.info("The seed number is set to {}".format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True


# ---------- dataloader의 worker마다 seed 고정시키는 함수 ----------
def _init_fn(worker_id):
    seed = args.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


# ---------- 데이터셋 로딩 ----------
if args.dataset == 'cifar10':
    args.num_classes = 10
    if args.distill in ['crd', 'crdst']:  # NCE 기반 KD이면 샘플 버전 로딩
        train_dataset, test_dataset = get_cifar10_dataloaders_sample(data_folder="./dataset/data/CIFAR10/", k=args.nce_k, mode=args.mode)
    else:
        train_dataset, test_dataset = get_cifar10_dataloaders(data_folder="./dataset/data/CIFAR10/")

elif args.dataset == 'cifar100':
    args.num_classes = 100
    if args.distill in ['crd', 'crdst']:
        train_dataset, test_dataset = get_cifar100_dataloaders_sample(data_folder="../data/CIFAR100/", k=args.nce_k, mode=args.mode)
    else:
        train_dataset, test_dataset = get_cifar100_dataloaders(data_folder="../data/CIFAR100/", is_instance=False)
else:
    raise NotImplementedError


# ---------- DataLoader 생성 ----------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           worker_init_fn=None if args.seed is None else _init_fn)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False,
                                          num_workers=args.num_workers)

printRed(f"dataset: {args.dataset}, num of training data (50,000): {len(train_dataset)}, number of testing data (10,000): {len(test_dataset)}")                                          


# ---------- 모델 클래스 선택 및 초기화 ----------
model_class = globals().get(args.arch)  # 문자열로 받은 클래스명 → 실제 클래스
model = model_class(args)
model.to(device)


# ---------- 모델 파라미터 수 출력 ----------
num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)
logging.info("The number of parameters : {}".format(num_total_params))


# ---------- pretrained weight 불러오기 ----------
if args.load_pretrain:
    trained_model = torch.load(args.pretrain_path)
    current_dict = model.state_dict()
    printRed("Pretrained full precision weights are initialized")
    logging.info("\nFollowing modules are initialized from pretrained model")
    log_string = ''
    for key in trained_model['model'].keys():
        if key in current_dict.keys():
            log_string += '{}\t'.format(key)
            current_dict[key].copy_(trained_model['model'][key])
    logging.info(log_string+'\n')
    model.load_state_dict(current_dict)
else:
    printRed("Not initialized by the pretrained full precision weights")

# 양자화 파라미터 초기화 (QConv 안의 lW, uW, lA, uA 등을 데이터 기반으로 설정함)
init_quant_model(model, train_loader, device, args.distill)

# EWGS 방식 또는 baseline(STE 기반)인 경우만 양자화 학습 스케줄러를 정의함
if args.quan_method == "EWGS" or args.baseline:
    define_quantizer_scheduler = True
else:
    define_quantizer_scheduler = False

# 모델 파라미터와 양자화 파라미터 분리해서 따로 관리함
if args.quan_method == "EWGS" or args.baseline:
    trainable_params = list(model.parameters())
    model_params = []
    quant_params = []
    for m in model.modules():
        if isinstance(m, QConv):  # QConv 계층이면 일반 파라미터 + 양자화 파라미터 따로 저장
            model_params.append(m.weight)
            if m.bias is not None:
                model_params.append(m.bias)
            if m.quan_weight:
                quant_params.append(m.lW)
                quant_params.append(m.uW)
            if m.quan_act:
                quant_params.append(m.lA)
                quant_params.append(m.uA)
                quant_params.append(m.lA_t)
                quant_params.append(m.uA_t)
            if m.quan_act or m.quan_weight:
                quant_params.append(m.output_scale)
            print("QConv", m)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):  # 일반 conv, linear 계층
            model_params.append(m.weight)
            if m.bias is not None:
                model_params.append(m.bias)
            print("nn", m)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):  # BN 계층
            if m.affine:  # affine=True일 때만 학습 파라미터 존재함
                model_params.append(m.weight)
                model_params.append(m.bias)

    # 파라미터 수 출력 및 검증
    print("# total params:", sum(p.numel() for p in trainable_params))
    print("# model params:", sum(p.numel() for p in model_params))
    print("# quantizer params:", sum(p.numel() for p in quant_params))
    logging.info("# total params: {}".format(sum(p.numel() for p in trainable_params)))
    logging.info("# model params: {}".format(sum(p.numel() for p in model_params)))
    logging.info("# quantizer params: {}".format(sum(p.numel() for p in quant_params)))

    # 총 파라미터 수가 일치하는지 검증
    if sum(p.numel() for p in trainable_params) != sum(p.numel() for p in model_params) + sum(p.numel() for p in quant_params):
        raise Exception('Mismatched number of trainable parmas')
else:
    raise NotImplementedError(f"Not implement {args.quan_method}!")

# Knowledge Distillation 사용하는 경우, teacher 모델 로드 및 distill 모듈 설정
if args.distill:
    model_class_t = globals().get(args.teacher_arch)
    model_t = model_class_t(args)
    model_t.to(device)
    model_t = utils.load_teacher_model(model_t, args.teacher_path)

    num_training_data = len(train_dataset)
    module_list, model_params, criterion_list = utils_distill.define_distill_module_and_loss(
        model, model_t, model_params, args, num_training_data, train_loader
    )

# 양자화 파라미터에 대한 옵티마이저 및 러닝레이트 스케줄러 정의
if define_quantizer_scheduler:
    if args.optimizer_q == 'SGD':
        optimizer_q = torch.optim.SGD(quant_params, lr=args.lr_q)
    elif args.optimizer_q == 'Adam':
        optimizer_q = torch.optim.Adam(quant_params, lr=args.lr_q)

    if args.lr_scheduler_q == "step":
        milestones_q = list(map(int, args.decay_schedule_q.split('-'))) if args.decay_schedule_q else [args.epochs+1]
        scheduler_q = torch.optim.lr_scheduler.MultiStepLR(optimizer_q, milestones=milestones_q, gamma=args.gamma)
    elif args.lr_scheduler_q == "cosine":
        scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=args.epochs, eta_min=args.lr_q_end)

# 모델 파라미터에 대한 optimizer 및 scheduler 정의
if args.optimizer_m == 'SGD':
    optimizer_m = torch.optim.SGD(model_params, lr=args.lr_m, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer_m == 'Adam':
    optimizer_m = torch.optim.Adam(model_params, lr=args.lr_m, weight_decay=args.weight_decay)

if args.lr_scheduler_m == "step":
    milestones_m = list(map(int, args.decay_schedule_m.split('-'))) if args.decay_schedule_m else [args.epochs+1]
    scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=milestones_m, gamma=args.gamma)
elif args.lr_scheduler_m == "cosine":
    scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, eta_min=args.lr_m_end)

# teacher 모델은 weight_decay 방지 위해 optimizer 이후에 추가함
if args.distill:
    module_list.append(model_t)
    module_list.cuda()
    criterion_list.cuda()

criterion = nn.CrossEntropyLoss()  # 기본 분류 손실 함수 정의
writer = SummaryWriter(args.log_dir)  # 텐서보드 writer 초기화

# 학습 관련 변수 초기화
total_iter = 0
best_acc = 0
acc_last5 = []
iterations_per_epoch = len(train_loader)
lambda_dict = {}
print(f"iterations_per_epoch: {iterations_per_epoch}")

# ============================ Epoch 단위 학습 루프 시작 ============================
for ep in range(args.epochs):
    if args.distill:
        for module in module_list:
            module.train()
        module_list[-1].eval()
        if args.distill == 'abound':
            module_list[1].eval()
        elif args.distill == 'factor':
            module_list[2].eval()
        criterion_cls, criterion_div, criterion_kd = criterion_list
        model_s = module_list[0]
        model_t = module_list[-1]
    else:
        model.train()

    # Hessian trace 기반 grad scale 업데이트
    if ep % args.update_every == 0 and ep != 0 and not args.baseline and args.use_hessian:
        print("update grade scales")

    writer.add_scalar('train/model_lr', optimizer_m.param_groups[0]['lr'], ep)
    if define_quantizer_scheduler:
        writer.add_scalar('train/quant_lr', optimizer_q.param_groups[0]['lr'], ep)

    # mini-batch loop
    for i, data in enumerate(train_loader):
        if args.distill in ["crd", "crdst"]:
            images, labels, index, contrast_idx = data
            index = index.to(device)
            contrast_idx = contrast_idx.to(device)
        else:
            images, labels = data
            index, contrast_idx = None, None

        images = images.to(device)
        labels = labels.to(device)

        optimizer_m.zero_grad()
        if define_quantizer_scheduler:
            optimizer_q.zero_grad()

        if args.quan_method == "EWGS":
            save_dict = {"iteration": total_iter, "writer": writer, "layer_num": None, "block_num": None, "conv_num": None, "type": None}
            if total_iter >= 2:
                for i in range(total_iter - 1):
                    lambda_dict[f"{i}"] = {}
            lambda_dict[f"{total_iter}"] = {}
        else:
            save_dict = None
            lambda_dict = None

        # forward (distillation 여부에 따라 다르게 동작)
        if args.distill:
            flatGroupOut = args.distill == 'crdst'
            preact = args.distill in ['abound']
            feat_s, block_out_s, logit_s = model_s(images, save_dict, lambda_dict, is_feat=True, preact=preact, flatGroupOut=flatGroupOut)
            with torch.no_grad():
                feat_t, block_out_t, logit_t = model_t(images, is_feat=True, preact=preact, flatGroupOut=flatGroupOut)
                feat_t = [f.detach() for f in feat_t]
        else:
            pred = model(images, save_dict, lambda_dict)

        # backward + loss 계산
        if args.distill:
            loss_cls = criterion_cls(logit_s, labels)
            loss_div = criterion_div(logit_s, logit_t)
            if args.distill == "crdst":
                loss_kd_crd, loss_kd_crdSt = utils_distill.get_loss_crdst(args, feat_s, feat_t, criterion_kd, index, contrast_idx, block_out_s, block_out_t)
                loss_total = args.kd_gamma * loss_cls + args.kd_alpha * loss_div + args.kd_beta * loss_kd_crd + args.kd_theta * loss_kd_crdSt 
            else:
                loss_kd = utils_distill.get_loss_kd(args, feat_s, feat_t, criterion_kd, module_list, index, contrast_idx)
                loss_total = args.kd_gamma * loss_cls + args.kd_alpha * loss_div + args.kd_beta * loss_kd

            # loss 기록
            writer.add_scalar('train/loss_cls', loss_cls.item(), total_iter)
            writer.add_scalar('train/loss_div', loss_div.item(), total_iter)
            writer.add_scalar('train/loss', loss_total.item(), total_iter)
            with open(os.path.join(args.log_dir,'loss.txt'), "a") as w:
                w.write(f"total_iter={total_iter}, loss_cls={loss_cls.item()}, loss_div={loss_div.item()}, loss_total={loss_total.item()}\n")
            if i == 0:
                printRed(f"gamma: {args.kd_gamma}, alpha: {args.kd_alpha}, kd_beta: {args.kd_beta}, kd_theta: {args.kd_theta}")
        else:
            loss_total = criterion(pred, labels)

        loss_total.backward()
        optimizer_m.step()
        if define_quantizer_scheduler:
            optimizer_q.step()

        total_iter += 1

    # epoch 종료 후 scheduler step 진행
    scheduler_m.step()
    if define_quantizer_scheduler:
        scheduler_q.step()

    # train & test accuracy 평가 및 텐서보드 기록
    with torch.no_grad():
        for phase, loader, tag in [(train_loader, "train"), (test_loader, "test")]:
            model.eval()
            correct, total = 0, 0
            for data in loader:
                images, labels = data[:2]
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images)
                _, predicted = torch.max(pred.data, 1)
                total += pred.size(0)
                correct += (predicted == labels).sum().item()
            acc = correct / total * 100
            writer.add_scalar(f'{tag}/acc', acc, ep)
            if tag == "test":
                test_acc = acc

    print(f"Current epoch: {ep:03d} \t Test accuracy: {test_acc:.2f} %")
    logging.info(f"Current epoch: {ep:03d}\t Test accuracy: {test_acc:.2f}%")

    # 체크포인트 저장
    torch.save({...}, os.path.join(args.log_dir,'checkpoint/last_checkpoint.pth'))
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({...}, os.path.join(args.log_dir,'checkpoint/best_checkpoint.pth'))
    if ep >= args.epochs - 5:
        acc_last5.append(test_acc)

    # QConv 계층에 대한 파라미터 로깅
    layer_num = 0
    for m in model.modules():
        if isinstance(m, QConv):
            layer_num += 1
            if args.QWeightFlag:
                writer.add_scalar(f"z_{layer_num}th_module/lW", m.lW.item(), ep)
                writer.add_scalar(f"z_{layer_num}th_module/uW", m.uW.item(), ep)
                writer.add_scalar(f"z_{layer_num}th_module/bkwd_scaleW", m.bkwd_scaling_factorW.item(), ep)
            if args.QActFlag:
                writer.add_scalar(f"z_{layer_num}th_module/lA", m.lA.item(), ep)
                writer.add_scalar(f"z_{layer_num}th_module/uA", m.uA.item(), ep)
                writer.add_scalar(f"z_{layer_num}th_module/bkwd_scaleA", m.bkwd_scaling_factorA.item(), ep)
            if args.QWeightFlag or args.QActFlag:
                writer.add_scalar(f"z_{layer_num}th_module/output_scale", m.output_scale.item(), ep)

# 최종 정확도 및 요약 출력
utils.test_accuracy(checkpoint_path_last, model, logging, device, test_loader)
utils.test_accuracy(checkpoint_path_best, model, logging, device, test_loader)
mean_last5 = round(np.mean(acc_last5), 2)
print(f"Average accuracy of the last 5 epochs: {mean_last5}, acc_last5: {acc_last5}\n")
logging.info(f"Average accuracy of the last 5 epochs: {mean_last5}, acc_last5: {acc_last5}\n")
print(f"Total time: {(time.time()-start_time)/3600:.2f}h")
logging.info(f"Total time: {(time.time()-start_time)/3600:.2f}h")
print(f"Save to {args.log_dir}")
logging.info(f"Save to {args.log_dir}")
