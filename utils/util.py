import torch


def get_parameter_number(net):
    """
    모델의 전체 파라미터 수와 학습 가능한 파라미터 수를 계산함
    + requires_grad=True인 파라미터들의 이름과 shape도 출력해줌
    """
    # 전체 파라미터 수 (numel(): 요소 개수 반환)
    total_num = sum(p.numel() for p in net.parameters())

    # 학습 가능한 파라미터 수 (requires_grad=True인 것만 카운트)
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # 어떤 파라미터가 학습 가능한지 이름과 shape 출력
    for name, para in net.named_parameters():
        if para.requires_grad:
            print(name, ": ", para.shape)

    # dict 형태로 반환
    return {"Total": total_num, "Trainable": trainable_num}


def transfer_model(pretrained_file, model):
    """
    사전학습된 weight를 현재 모델에 옮겨주는 함수
    - 특정 key(prefix가 없는 경우)를 'basemodel.xxx' 형태로 맞춰서 로딩
    """
    # pretrained 파일에서 state_dict 추출
    pretrained_dict = torch.load(pretrained_file).state_dict()

    # 현재 모델의 state_dict 가져오기
    model_dict = model.state_dict()

    # 우리가 옮기려는 건 basemodel에 해당하는 weight들이라 key만 추림
    model_keys = model.basemodel.state_dict().keys()

    # key 이름을 'basemodel.xxx' 형태로 맞춰줌 (필요한 것만 필터링)
    pretrained_dict = transfer_state_dict(pretrained_dict, model_keys)

    # 기존 모델 dict에 pretrained weight 덮어쓰기
    model_dict.update(pretrained_dict)

    # 모델에 적용
    model.load_state_dict(model_dict)

    return model


def transfer_state_dict(pretrained_dict, model_keys):
    """
    key 이름을 'basemodel.xxx' 형식으로 바꿔주는 함수
    → 현재 모델의 구조에 맞게 key prefix를 붙여주는 역할
    """
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_keys:
            # key가 일치하면 'basemodel.' prefix 붙여서 새 dict에 저장
            state_dict["basemodel." + k] = v
        else:
            # 없는 키는 경고 출력만 하고 무시
            print("Missing key(s) in state_dict :{}".format(k))

    return state_dict
