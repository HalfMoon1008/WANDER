from torch.utils.data import DataLoader

from dataloader.mosidata import CMUData
from dataloader.fooddata import Food101


def getdataloader(dataset, data_path, batch_size):
    """
    데이터셋 이름에 따라 DataLoader를 구성해서 반환해주는 함수
    - mosi, food 두 가지 지원
    - 학습/검증/테스트 세 가지 split에 대해 DataLoader를 만들어서 dict 형태로 반환
    - multimodal 입력을 위한 orig_dim, t_dim 값도 함께 반환 (mosi만 해당)

    :param dataset: "mosi" 또는 "food"
    :param data_path: 데이터셋 루트 경로
    :param batch_size: 배치 크기
    :return: (dict of DataLoader, orig_dim, t_dim)
    """

    if dataset == "mosi":
        # CMU-MOSI multimodal 데이터셋 로딩
        data = {
            "train": CMUData(data_path, "train"),
            "valid": CMUData(data_path, "valid"),
            "test": CMUData(data_path, "test"),
        }

        # 테스트 셋에서 원본 차원 정보 추출 (예: 텍스트/오디오/비디오 차원)
        orig_dim = data["test"].get_dim()
        t_dim = data["test"].get_tim()

        # 각 split에 대해 DataLoader 생성 (병렬 처리 위해 num_workers=8)
        dataLoader = {
            ds: DataLoader(data[ds], batch_size=batch_size, num_workers=8)
            for ds in data.keys()
        }

    elif dataset == "food":
        # Food101 이미지-텍스트 분류 데이터셋 로딩
        data = {
            "train": Food101(mode="train", dataset_root_dir=data_path),
            "valid": Food101(mode="test", dataset_root_dir=data_path),
            "test": Food101(mode="test", dataset_root_dir=data_path),
        }

        # food 데이터셋은 multimodal 차원이 없으므로 None으로 설정
        orig_dim, t_dim = None, None

        # DataLoader 구성
        dataLoader = {
            ds: DataLoader(data[ds], batch_size=batch_size, num_workers=8)
            for ds in data.keys()
        }

    # DataLoader, 차원 정보 함께 반환
    return dataLoader, orig_dim, t_dim
