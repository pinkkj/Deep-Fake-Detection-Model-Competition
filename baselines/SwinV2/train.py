from datasets import load_dataset, DownloadConfig
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import default_data_collator
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import argparse

import random
import cv2
from PIL import Image, ImageFile
import io
ImageFile.LOAD_TRUNCATED_IMAGES = True # 깨진/불완전한 이미지도 로딩 허용 (LAION 스타일 데이터 대비)

"""
------------------------------
이미지 블러 측정 함수
------------------------------
"""
def estimate_blur_laplacian(img_np):
    """
    Laplacian variance로 이미지의 ‘선명도/블러 정도’를 추정.
    값이 클수록 선명, 작을수록 블러가 큰 이미지.
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

"""
------------------------------
Fake 이미지에만 적용하는 '가벼운 열화(Degradation)' 함수
 - 실학습 데이터의 해상도/블러 분포에 맞춰주기 위한 목적
------------------------------
"""
def degrade_image_to_match_laion5(img_pil, real_blur_vals, real_res_vals,
                                  noise_var=0.0005, jpeg_quality_range=(70, 95), seed=None):
    """
    real_blur_vals: 실제 학습 데이터(real 이미지)의 Laplacian blur 분포 (np.load한 값)
    real_res_vals : 실제 학습 데이터 해상도(H, W) 분포
    noise_var     : 추가할 가우시안 노이즈 분산 (기본 0.0005)
    jpeg_quality_range: JPEG 압축 품질 범위
    seed          : 재현 가능성을 위한 랜덤 시드 (옵션)

    [검증 환경에서 수정 가능]
    - noise_var, jpeg_quality_range 값은 augmentation 강도 조절을 위해 변경 가능하지만,
      성능 재현을 위해 검증 시에는 대회에서 사용한 설정을 그대로 유지하는 것이 바람직.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    """
    === Step 1: 해상도 매칭 (20% 확률로 적용) ===
    real_res_vals에서 임의의 (H, W)를 뽑아, 현재 이미지의 overall scale을 맞춰줌
    """
    if random.random() < 0.2:
        target_h, target_w = random.choice(real_res_vals)
        orig_w, orig_h = img_pil.size
        orig_area = orig_w * orig_h
        target_area = target_h * target_w
        scale = (target_area / orig_area) ** 0.5
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
    img_np = np.array(img_pil)
    """
    === Step 2: Blur 매칭 (20% 확률) ===
    현재 blur_val이 타겟 blur보다 충분히 클 때에만 Gaussian Blur를 살짝 걸어줌
    """
    if random.random() < 0.2:
        target_blur = np.random.choice(real_blur_vals)
        blur_val = estimate_blur_laplacian(img_np)
        if blur_val > target_blur * 1.2:
            img_np = cv2.GaussianBlur(img_np, (0, 0), sigmaX=0.3, sigmaY=0.3)
    """
    === Step 3: 가우시안 노이즈 추가 (20% 확률) ===
    """
    if random.random() < 0.2:
        sigma = int(255 * (noise_var ** 0.5))
        if sigma > 0:
            noise = np.zeros_like(img_np, dtype=np.int16)
            cv2.randn(noise, 0, sigma)             # in‑place Gaussian noise
            img_np = cv2.add(img_np.astype(np.int16), noise, dtype=cv2.CV_8U)
    """   
    === Step 4: JPEG 압축으로 인한 아티팩트 추가 (20% 확률) ===
    """ 
    if random.random() < 0.2:
        quality = np.random.randint(*jpeg_quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img_np, encode_param)
        img_np = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_np)

"""
------------------------------
메인 학습 함수
------------------------------
"""
def main(args):
    # W&B 프로젝트 이름 설정
    # [검증 환경에서 변경 가능]
    # - WANDB를 사용하지 않는 환경에서는 이 라인은 무시되거나, WANDB_DISABLED=true로 비활성화하도록 안내 가능
    os.environ["WANDB_PROJECT"] = "SwinOpenFake"
    # HF datasets / 모델 캐시 폴더 설정
    os.makedirs(args.cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    # SwinV2 이미지 프로세서 / 분류 모델 로드
    # Hugging Face Hub에서 microsoft/swinv2-base-patch4-window16-256를 로드
    processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window16-256", cache_dir=args.cache_dir, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window16-256", cache_dir=args.cache_dir)
    
    # 클래스 수를 2개(Real/Fake)로 재설정하고 최종 분류 레이어 교체
    model.num_labels = 2
    model.config.num_labels = 2
    model.classifier = torch.nn.Linear(model.swinv2.num_features, model.num_labels)
    # [검증 환경에서 변경 가능]
    # - 기본은 CUDA 사용. GPU가 없을 경우 "cpu"로 변경 가능.
    #   단, 대회 성능 재현은 GPU 기준.
    model.to("cuda")
    
    # Real train 통계(blur, resolution) 로딩
    # ../real_train_stats.npz 는 학습 시 사용한 통계 파일로, 검증 환경에도 같이 제공되어야 함.
    real_train_stats = np.load('../real_train_stats.npz')
    real_blur_vals = real_train_stats['blur_vals']
    real_res_vals  = real_train_stats['res_vals']

    # ------------------------------
    # OpenFake 데이터셋 로드
    # ------------------------------
    # streaming=False : 전체 데이터를 로컬에 다운로드하여 사용
    train_data = load_dataset(
    "ComplexDataLab/OpenFake", split="train",
    streaming=False, download_config=DownloadConfig(cache_dir=args.cache_dir, resume_download=True, max_retries=10)
    )
    eval_data  = load_dataset(
    "ComplexDataLab/OpenFake", split="test",
    streaming=False, download_config=DownloadConfig(cache_dir=args.cache_dir, resume_download=True, max_retries=10)
    )

    # ------------------------------
    # 전처리 함수 (Train)
    #  - Fake 라벨(1)에만 degrade_image_to_match_laion5 적용
    # ------------------------------
    def preprocess_train(example):
        image = example["image"]

        # 이미지 타입을 PIL.Image(RGB)로 통일
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, dict):
                if "bytes" in image and image["bytes"] is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                elif "path" in image and image["path"]:
                    image = Image.open(image["path"])
                else:
                    raise ValueError(f"Unsupported image dict keys: {image.keys()}")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        if image.mode != "RGB":
            image = image.convert("RGB")
        raw_label = example["label"]
        # 라벨을 정수형 0/1로 매핑 (0=real, 1=fake)
        if isinstance(raw_label, str):
            label = 0 if raw_label.lower() == "real" else 1
        else:
            label = int(raw_label)
        # Fake 이미지에만 degradation 적용
        if label == 1:
            image = degrade_image_to_match_laion5(
                image, real_blur_vals, real_res_vals,
                seed=args.seed if hasattr(args, "seed") else None
            )
        inputs = processor(image, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"].squeeze(0), "label": label}

    # ------------------------------
    # 전처리 함수 (Eval)
    #  - 검증/테스트 시에는 augmentation 없이 원본만 사용
    # ------------------------------
    def preprocess_eval(example):
        image = example["image"]
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, dict):
                if "bytes" in image and image["bytes"] is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                elif "path" in image and image["path"]:
                    image = Image.open(image["path"])
                else:
                    raise ValueError(f"Unsupported image dict keys: {image.keys()}")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        if image.mode != "RGB":
            image = image.convert("RGB")
        raw_label = example["label"]

        if isinstance(raw_label, str):
            label = 0 if raw_label.lower() == "real" else 1
        else:
            label = int(raw_label)

        inputs = processor(image, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"].squeeze(0), "label": label}

    # ------------------------------
    # map을 이용한 일괄 전처리
    # ------------------------------
    # num_proc: 멀티프로세스로 전처리 병렬화
    # [검증 환경에서 변경 가능]
    # - CPU 코어가 적거나 Windows 기반일 경우 num_proc=1로 줄여야 할 수 있음.
    train_data = train_data.map(preprocess_train, num_proc=4)
    eval_data  = eval_data.map(preprocess_eval,  num_proc=2)

    # ------------------------------
    # Metric 계산 함수
    # ------------------------------
    def compute_metrics(pred):
        logits = pred.predictions
        preds = logits.argmax(-1)
        labels = pred.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        auc_roc = roc_auc_score(labels, preds)
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
        }
    # ------------------------------
    # max_steps 설정
    # ------------------------------
    # 600000 // batch_size * num_epochs 형태로 최대 스텝 수를 제한.
    # 실제 데이터셋 크기와는 별개로 upper bound를 두는 역할.
    # [검증 환경에서 변경 가능]
    # - 필요시 제거하거나 값 조정 가능하나, 대회 학습 재현 시에는 동일 공식 유지 권장.
    max_steps = 600000 // args.batch_size * args.num_epochs

    # ------------------------------
    # HuggingFace TrainingArguments 설정
    # ------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir, # 체크포인트 저장 경로
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=20,  # 저장할 체크포인트 최대 개수
        logging_steps=100,
        eval_strategy="steps",  # 일정 step마다 evaluation
        eval_steps=500,
        metric_for_best_model="f1",
        greater_is_better=True,
        max_steps=max_steps,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        run_name="swinv2-finetuned-openfake",
        report_to="wandb",  # [검증 환경에서 변경 가능] "none"으로 바꾸면 로깅 비활성화
    )

    # ------------------------------
    # Trainer 초기화
    # ------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    return trainer, eval_data

# ------------------------------
# CLI 진입점
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on OpenFake dataset")
    # 출력 디렉토리, epoch, batch size 등 하이퍼파라미터
    parser.add_argument("--output_dir", type=str, default="./swinv2-finetuned-openfake", help="Output directory for model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer")
    parser.add_argument("--num_workers", type=int, default=4,help="DataLoader worker processes")
    parser.add_argument("--cache_dir", type=str, default='.cache', help="Cache directory for datasets and models")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory or checkpoint name to resume training from"
    )
    # [옵션] seed를 커맨드라인에서 받을 수도 있음 (현재 코드에서는 args.seed 사용 시에만 참조)
    # parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    trainer, eval = main(args)
    # 학습 시작 (resume_from_checkpoint 옵션 지원)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    #trainer.train(resume_from_checkpoint=False)
    #trainer._load_from_checkpoint(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # 학습 완료 후 평가
    eval_results = trainer.evaluate(eval_dataset=eval)
    print(f"Evaluation results: {eval_results}")