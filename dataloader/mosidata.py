from torch.utils.data.dataset import Dataset
import pickle
import torch


class CMUData(Dataset):
    def __init__(self, data_path, split):
        """
        data_path: 피클 파일 (.pkl) 경로
        split: 'train' / 'valid' / 'test' 중 하나
        """

        # pickle 파일 열어서 전체 데이터 로드
        with open(data_path, 'rb') as file:
            data = pickle.load(file)

        # split(train/valid/test) 별 데이터 가져오기
        self.data = data[split]
        self.split = split

        # 각 modality의 차원 크기 저장
        # 예: [text_dim, audio_dim, vision_dim]
        self.orig_dims = [
            self.data['text'][0].shape[1],     # 텍스트 feature dimension
            self.data['audio'][0].shape[1],    # 오디오 feature dimension
            self.data['vision'][0].shape[1]    # 비디오 feature dimension
        ]

    def get_dim(self):
        """
        각 modality의 feature 차원 (input dim)을 반환함
        예: [300, 74, 35] 같이 나올 수 있음
        """
        return self.orig_dims

    def get_tim(self):
        """
        각 modality의 시퀀스 길이(time step)를 반환함
        예: [seq_len_text, seq_len_audio, seq_len_vision]
        → 보통 한 sample 기준으로 고정되어 있음
        """
        return [
            self.data['text'][0].shape[0],
            self.data['audio'][0].shape[0],
            self.data['vision'][0].shape[0]
        ]

    def __len__(self):
        """
        전체 샘플 개수 반환 (보통 utterance 단위)
        """
        return self.data['audio'].shape[0]

    def __getitem__(self, idx):
        """
        idx 번째 샘플의 각 modality 텐서를 반환함
        - text: (seq_len, dim)
        - audio: (seq_len, dim)
        - vision: (seq_len, dim)
        - labels: 회귀 레이블 (float 값)

        float()로 변환해서 모델 입력에 바로 쓸 수 있게 만듦
        """
        return {
            'audio': torch.tensor(self.data['audio'][idx]).float(),
            'vision': torch.tensor(self.data['vision'][idx]).float(),
            'text': torch.tensor(self.data['text'][idx]).float(),
            'labels': torch.tensor(self.data['regression_labels'][idx]).float(),
        }
