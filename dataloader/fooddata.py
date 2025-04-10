import random
import torch
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
from torch.utils.data import Dataset
from os.path import join
import json

# 클래스 이름 리스트 (총 101개 음식 클래스)
CLASS_NAME = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheese_plate",
    "cheesecake",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "escargots",
    "falafel",
    "filet_mignon",
    "fish_and_chips",
    "foie_gras",
    "french_fries",
    "french_onion_soup",
    "french_toast",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "huevos_rancheros",
    "hummus",
    "ice_cream",
    "lasagna",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "macaroni_and_cheese",
    "macarons",
    "miso_soup",
    "mussels",
    "nachos",
    "omelette",
    "onion_rings",
    "oysters",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "peking_duck",
    "pho",
    "pizza",
    "pork_chop",
    "poutine",
    "prime_rib",
    "pulled_pork_sandwich",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed_salad",
    "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna_tartare",
    "waffles",
]

TEXT_MAX_LENGTH = 50  # BERT tokenizer로 자를 최대 길이
MIN_FREQ = 3          # 사용은 안 됨 (아마 다른 곳에서 쓰는 global 상수로 선언만 해둠)
NUMBER_OF_SAMPLES_PER_CLASS = None  # (현재 사용 X)


# 🔹 1. Datapool 클래스 (전체 데이터셋의 관리 구조)
class Datapool(Dataset):
    def __init__(self, all_ids, mode):
        self.all_ids = all_ids
        self.mode = mode

        # ID 중복 체크 (중복되면 에러 발생시킴)
        assert len(list(set(self.all_ids))) == len(self.all_ids), "dataset has duplicated ids"

        # labeled / unlabeled split 초기화
        self.unlabeled_ids = self.all_ids.copy()  # 처음엔 전부 unlabeled 상태
        self.labeled_ids = []

        # 기본 sample_ids 설정 (test/val용일 경우 전부 사용)
        self.sample_ids = self.all_ids.copy()

    def initialize(self, query_budget: int):
        """
        Active learning 초기에 라벨링할 샘플을 random으로 선택하는 함수
        query_budget만큼 뽑고 나머지를 unlabeled로 유지
        """
        self.labeled_ids = self.unlabeled_ids[:query_budget]
        self.unlabeled_ids = [id for id in self.all_ids if id not in self.labeled_ids]

    def query_for_label(self, queried_ids: list):
        """
        Query 전략이 선택한 샘플들을 라벨링된 리스트에 추가
        """
        self.labeled_ids += queried_ids
        self.unlabeled_ids = [id for id in self.all_ids if id not in self.labeled_ids]
        assert len(self.labeled_ids) + len(self.unlabeled_ids) == len(self.all_ids)

    def query(self):
        """
        쿼리용 데이터셋 설정 (라벨링 안 된 데이터만 사용함)
        """
        self.mode = "query"
        print("dataset for querying")
        self.sample_ids = self.unlabeled_ids

    def train(self):
        """
        학습용 데이터셋 설정 (라벨링 된 데이터만 사용함)
        """
        self.mode = "train"
        print("dataset for training")
        self.sample_ids = self.labeled_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # 추상 클래스라서 구현하지 않음. Food101에서 override함
        pass


# 🔹 2. Food101 클래스 (실제 food101 데이터셋에 대한 구체 구현)
class Food101(Datapool):
    def __init__(
        self,
        mode="train",
        dataset_root_dir=r"data/food101/",
    ):
        self.dataset_root_dir = dataset_root_dir
        self.mode = mode
        assert self.mode in ["train", "dev", "test"]  # mode 유효성 체크

        # JSON에서 데이터 로드 → ID별로 dict 구성
        with open(join(dataset_root_dir, f"{mode}.json")) as file:
            data_list = json.load(file)
            self.data = {x["id"]: x for x in data_list}

        # 전체 샘플 id 목록 생성 후 섞기
        self.all_ids = list(self.data.keys())
        random.Random(0).shuffle(self.all_ids)  # seed 고정

        # 부모 클래스(Datapool) 초기화
        super(Food101, self).__init__(self.all_ids, self.mode)

        # Augmentation 설정
        color_distort_strength = 0.5
        color_jitter = transforms.ColorJitter(
            brightness=0.8 * color_distort_strength,
            contrast=0.8 * color_distort_strength,
            saturation=0.8 * color_distort_strength,
            hue=0.2 * color_distort_strength,
        )
        gaussian_kernel_size = 21

        # 학습용 transform (augmentation 포함)
        self.train_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=gaussian_kernel_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # 검증/테스트용 transform (augmentation 없음)
        self.val_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # 텍스트 관련 설정
        self.sentence_max_len = TEXT_MAX_LENGTH
        self.tokenizer = BertTokenizer.from_pretrained("pre-trained BERT")  # 모델 이름은 수정 필요할 수도 있음

    def load_bert_tokens(self, sample_id):
        """
        주어진 샘플의 텍스트 token을 BERT tokenizer로 인코딩함
        반환 형식은 tensor dict (input_ids, attention_mask 등)
        """
        text_tokens = " ".join(self.data[sample_id]["text_tokens"])
        text_input = self.tokenizer(
            text_tokens,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.sentence_max_len,
            truncation=True,
            return_tensors="pt",
        )
        # (1, L) → (L,)로 squeeze
        for k, v in text_input.items():
            text_input[k] = v.squeeze(0)
        return text_input

    def load_image(self, sample_id):
        """
        이미지 파일을 불러오고, 적절한 transform 적용
        """
        image_path = join(self.dataset_root_dir, self.data[sample_id]["img_path"])
        with open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.mode == "train":
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)
        return image

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """
        하나의 샘플을 불러와서 (text_input, image, label) 반환
        text_input: BERT input
        image: transform된 이미지 tensor
        label: 정수형 클래스 index (0~100)
        """
        sample_id = self.sample_ids[idx]
        text_input = self.load_bert_tokens(sample_id)
        class_name = self.data[sample_id]["label"]
        image = self.load_image(sample_id)
        label = torch.tensor(CLASS_NAME.index(class_name), dtype=torch.long)
        return text_input, image, label
