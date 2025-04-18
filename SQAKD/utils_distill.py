import torch.nn as nn
import torch

import sys

from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from distiller_zoo import SimilarityTransfer
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import FactorTransfer, FSP, NSTLoss
from crd.criterion import CRDLoss

from utils import printRed
 
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn

# Knowledge Distillation을 위한 모듈들과 loss 함수들을 정의함
def define_distill_module_and_loss(model_s, model_t, model_params, args, n_data, train_loader):
    printRed("Define distillation modules and loss terms")

    # crdst 방식에서는 block_output을 비교하므로 flatGroupOut 설정해줘야 함
    flatGroupOut = True if args.distill == 'crdst' else False

    # dummy 입력으로 teacher와 student의 feature shape을 알아냄
    data = torch.randn(2, 3, 32, 32).cuda()
    model_t.eval()
    model_s.eval()
    feat_t, block_out_t, _ = model_t(data, is_feat=True, flatGroupOut=flatGroupOut)
    feat_s, block_out_s, _ = model_s(data, is_feat=True, flatGroupOut=flatGroupOut)

    # nn.ModuleList로 module_list 생성, 항상 student 모델은 포함함
    module_list = nn.ModuleList([])
    module_list.append(model_s)

    # 기본 classification loss
    criterion_cls = nn.CrossEntropyLoss()
    # KL divergence를 이용한 기본 KD loss (logit level)
    criterion_div = DistillKL(args.kd_T)

    # 아래부터 distillation 방식별로 KD loss 정의 및 필요한 모듈 추가함

    if args.distill == 'kd':
        # 기본적인 soft-label KD 방식
        criterion_kd = DistillKL(args.kd_T)

    elif args.distill == 'crd':
        # Contrastive Representation Distillation
        args.s_dim = feat_s[-1].shape[1]  # student 마지막 feature 차원
        args.t_dim = feat_t[-1].shape[1]  # teacher 마지막 feature 차원
        args.n_data = n_data              # 전체 학습 데이터 수

        # contrastive loss 객체 생성
        criterion_kd = CRDLoss(args)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)

        # student/teacher 임베딩 파라미터를 학습 대상에 포함시킴
        for name, param in criterion_kd.embed_s.named_parameters():
            model_params.append(param)
        for name, param in criterion_kd.embed_t.named_parameters():
            model_params.append(param)

    elif args.distill == 'crdst':
        # Contrastive + Structural Transfer (layer-wise)
        similarity_transfer = SimilarityTransfer(args.st_method, args.arch)
        criterion_kd = nn.ModuleList([])
        criterion_kd.append(similarity_transfer)

        for i in range(len(feat_s)):
            # 마지막 layer는 student 기준으로 shape 맞춤
            if i < len(feat_s)-1:
                args.s_dim = feat_t[i].shape[1]
            else:
                args.s_dim = feat_s[i].shape[1]
            args.t_dim = feat_t[i].shape[1]
            args.n_data = n_data

            # 각 layer마다 CRD loss를 따로 정의함
            criterion_kd_single = CRDLoss(args)
            module_list.append(criterion_kd_single.embed_s)
            module_list.append(criterion_kd_single.embed_t)
            criterion_kd.append(criterion_kd_single)

            for name, param in criterion_kd_single.embed_s.named_parameters():
                model_params.append(param)
            for name, param in criterion_kd_single.embed_t.named_parameters():
                model_params.append(param)

    elif args.distill == 'hint':
        # FitNet 방식 (intermediate feature regression)
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[args.hint_layer].shape, feat_t[args.hint_layer].shape)
        module_list.append(regress_s)
        for name, param in regress_s.named_parameters():
            print(name, param.shape)
            model_params.append(param)

    elif args.distill == 'attention':
        # Attention Transfer 방식 (중간 attention map 비교)
        criterion_kd = Attention()

    elif args.distill == 'nst':
        # Neuron Selectivity Transfer (activation distribution)
        criterion_kd = NSTLoss()

    elif args.distill == 'similarity':
        # Similarity-Preserving KD (pairwise distance 유지)
        criterion_kd = Similarity()

    elif args.distill == 'rkd':
        # Relational Knowledge Distillation (거리 + 각도)
        criterion_kd = RKDLoss()

    elif args.distill == 'correlation':
        # Correlation Congruence 방식
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], args.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], args.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        for name, param in embed_s.named_parameters():
            model_params.append(param)
        for name, param in embed_t.named_parameters():
            model_params.append(param)

    elif args.distill == 'vid':
        # Variational Information Distillation
        s_n = [f.shape[1] for f in feat_s[1:-1]]  # 중간 feature 차원
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList([VIDLoss(s, t, t) for s, t in zip(s_n, t_n)])
        for name, param in criterion_kd.named_parameters():
            model_params.append(param)

    elif args.distill == 'factor':
        # Factor Transfer 방식
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)

        # 사전 학습 단계 (paraphraser를 MSE로 학습)
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, args)

        # 본 학습 단계 (translator 학습)
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        for name, param in translator.named_parameters():
            model_params.append(param)

    elif args.distill == 'fsp':
        # Flow of Solution Procedure 방식
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)

        # teacher의 feature 흐름을 student에 전달하기 위한 사전 학습
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())  # student의 feature 모듈 반환
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, args)

        # 이후 FSP loss는 classification과 함께 사용됨 (loss 자체는 위에서 초기화 완료)
        pass

    else:
        raise NotImplementedError(args.distill)

    # 최종적으로 loss 목록 정의
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss (soft label)
    criterion_list.append(criterion_kd)     # distillation-specific loss

    return module_list, model_params, criterion_list

# student와 teacher feature를 받아서 distillation loss를 계산하는 함수
def get_loss_kd(args, feat_s, feat_t, criterion_kd, module_list, index, contrast_idx):

    if args.distill == 'kd':
        # 기본적인 KD 방식 (logit만 쓰는 경우), 이미 KL div가 계산되므로 별도 loss 없음
        loss_kd = 0

    elif args.distill == 'crd':
        # 마지막 feature를 뽑아서 CRD loss 계산함
        f_s = feat_s[-1]
        f_t = feat_t[-1]
        loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)

    elif args.distill == 'hint':
        # student feature를 regression layer에 통과시킨 후 teacher와 비교함
        f_s = module_list[1](feat_s[args.hint_layer])
        f_t = feat_t[args.hint_layer]
        loss_kd = criterion_kd(f_s, f_t)

    elif args.distill == 'attention':
        # 중간 feature들의 attention map을 비교함
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_kd = sum(criterion_kd(g_s, g_t))

    elif args.distill == 'nst':
        # NST 방식도 attention과 유사하게 중간 layer의 activation 분포 비교
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_kd = sum(criterion_kd(g_s, g_t))

    elif args.distill == 'similarity':
        # 마지막에서 두 번째 layer 기준으로 pairwise similarity 보존 여부 확인함
        g_s = [feat_s[-2]]
        g_t = [feat_t[-2]]
        loss_kd = sum(criterion_kd(g_s, g_t))

    elif args.distill in ['rkd', 'pkt']:
        # 거리 기반(RKD) 또는 probability 기반(PKT) 방식은 마지막 feature 사용
        f_s = feat_s[-1]
        f_t = feat_t[-1]
        loss_kd = criterion_kd(f_s, f_t)

    elif args.distill == 'kdsvd':
        # attention과 유사하지만, SVD 기반 loss 사용
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_kd = sum(criterion_kd(g_s, g_t))

    elif args.distill == 'correlation':
        # feature를 embedding 후 correlation loss 계산
        f_s = module_list[1](feat_s[-1])
        f_t = module_list[2](feat_t[-1])
        loss_kd = criterion_kd(f_s, f_t)

    elif args.distill == 'vid':
        # 각 중간 layer마다 VID loss 적용함
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
        loss_kd = sum(loss_group)

    elif args.distill == 'abound':
        # abound는 init 단계에서만 loss를 쓰므로 여기서는 없음
        loss_kd = 0

    elif args.distill == 'fsp':
        # FSP도 마찬가지로 init 훈련에서만 loss를 사용함
        loss_kd = 0

    elif args.distill == 'factor':
        # translator로 student feature 변환 후 paraphraser로 teacher와 비교함
        factor_s = module_list[1](feat_s[-2])
        factor_t = module_list[2](feat_t[-2], is_factor=True)
        loss_kd = criterion_kd(factor_s, factor_t)

    else:
        raise NotImplementedError(args.distill)

    return loss_kd

# crdst 방식 전용 KD loss 계산 함수
def get_loss_crdst(args, feat_s, feat_t, criterion_kd, index, contrast_idx, block_out_s, block_out_t):
    assert args.distill == 'crdst'  # 안전하게 확인함

    # block_out 기반으로 유사한 feature pair를 찾음 (구조 비교용)
    layer_pair_list = criterion_kd[0](block_out_s, block_out_t)

    loss_kd_crdSt_list = []

    # 첫 번째 layer는 별도 계산 (feat_s[0], feat_t[0])
    f0_s = feat_s[0]
    f0_t = feat_t[0]
    loss_kd_crdSt_list.append(criterion_kd[1](f0_s, f0_t, index, contrast_idx))

    # 나머지 layer pair에 대해 CRD loss 계산
    for i in range(2, len(layer_pair_list)+2): 
        f_s, f_t = layer_pair_list[i-2]
        loss_kd_crdSt_list.append(criterion_kd[i](f_s, f_t, index, contrast_idx))

    # 마지막 feature는 CRD loss로 따로 계산 (feat_s[-1], feat_t[-1])
    f_s = feat_s[-1]
    f_t = feat_t[-1]
    loss_kd_crd = criterion_kd[-1](f_s, f_t, index, contrast_idx)

    # 전체 구조 기반 loss 합산
    loss_kd_crdSt = sum(loss_kd_crdSt_list)

    return loss_kd_crd, loss_kd_crdSt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# abound, factor, fsp distillation을 위한 초기 훈련 루틴 정의
def init(model_s, model_t, init_modules, criterion, train_loader, args):
    printRed("Init modules for abound, factor, fsp")

    model_t.eval()
    model_s.eval()
    init_modules.train()  # paraphraser, translator 등만 학습함

    if torch.cuda.is_available():
        model_s.cuda()
        model_t.cuda()
        init_modules.cuda()
        cudnn.benchmark = True

    # factor에서만 lr 0.01 고정, 나머지는 일반 학습률 사용
    model_name = args.arch.split("_")[0]
    if model_name in ['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                      'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2'] and args.distill == 'factor':
        lr = 0.01
    else:
        lr = args.lr_m

    optimizer = optim.SGD(init_modules.parameters(),
                          lr=lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # 시간과 loss 측정을 위한 meter 정의
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(1, args.init_epochs + 1):
        batch_time.reset()
        data_time.reset()
        losses.reset()
        end = time.time()

        for idx, data in enumerate(train_loader):
            if args.distill in ['crd', 'crdst']:
                input, target, index, contrast_idx = data
            else:
                input, target = data

            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                if args.distill in ['crd', 'crdst']:
                    contrast_idx = contrast_idx.cuda()

            # student와 teacher 모두 forward 수행 (gradient는 student만)
            preact = (args.distill == 'abound')
            feat_s, _, _ = model_s(input, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, _, _ = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            # distill 방식별로 loss 계산
            if args.distill == 'abound':
                g_s = init_modules[0](feat_s[1:-1])
                g_t = feat_t[1:-1]
                loss = sum(criterion(g_s, g_t))

            elif args.distill == 'factor':
                f_t = feat_t[-2]
                _, f_t_rec = init_modules[0](f_t)  # paraphraser의 reconstruction
                loss = criterion(f_t_rec, f_t)

            elif args.distill == 'fsp':
                loss = sum(criterion(feat_s[:-1], feat_t[:-1]))

            else:
                raise NotImplemented('Not supported in init training: {}'.format(args.distill))

            losses.update(loss.item(), input.size(0))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        # 매 epoch마다 로그 출력
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
               epoch, args.init_epochs, batch_time=batch_time, losses=losses))
        sys.stdout.flush()
