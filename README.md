# VerifyAI – SwinV2 기반 딥페이크 탐지 (OpenFake)
이 저장소는 「2025 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회」 참가팀
VerifyAI가 제출한 딥페이크 이미지 탐지 모델의 코드와 환경 정의를 포함합니다.

모델은 Microsoft SwinV2 Base([SwinV2_Base](https://huggingface.co/microsoft/swinv2-base-patch4-window16-256?utm_source=chatgpt.com)) 비전 트랜스포머를 기반으로 하고,
최신 딥페이크 벤치마크 데이터셋인 OpenFake를 사용해 학습되었습니다.

# 1. 리포지토리 구성
예시 디렉터리 구조는 다음과 같습니다.
```text
.
├─ baselines/
|   └─ real_train_stats.npz  # real 이미지 해상도/블러 통계
│   └─ SwinV2/
│       ├─ train.py          # SwinV2 학습 + 평가 메인 스크립트
│       ├─ train.sh          # (선택) SLURM 클러스터용 학습 스크립트
│       ├─ deep_env.yaml     # Conda 가상환경 정의
│       └─ README.md         # SwinV2 전용 README
└─ README.md                 # (현재 파일) 대회 제출용 메인 설명
```
이 README에서는 SwinV2 기반 최종 제출 모델을 중심으로 설명합니다.
# 2. 사용 모델 개요
- 기본 백본 모델
    - 이름: microsoft/swinv2-base-patch4-window16-256
    - 종류: ImageNet-1k로 사전학습된 비전 트랜스포머 (SwinV2 Base) [SwinV2 Base](https://huggingface.co/microsoft/swinv2-base-patch4-window16-256?utm_source=chatgpt.com)
    - 라이선스: Apache-2.0 
- 파인튜닝 방식
    - SwinV2 backbone 위에 2-class linear classifier head를 올려
        - 0 = real
        - 1 = fake (synthetic)
    - Hugging Face AutoImageProcessor로 입력 이미지를 전처리합니다.

# 3. 데이터셋 설명(참고: [openfake_an open dataset and platform toward real World deepfake detection](https://pinkkj.github.io/posts/CV-OpenFake_An-Open-Dataset-and-Platform-Toward-Real-world-Deepfake-Detection/)
## 3.1 OpenFake 전체 개요
본 프로젝트는 Hugging Face Hub의 ComplexDataLab/OpenFake 데이터셋을 학습/평가에 사용합니다. [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake?utm_source=chatgpt.com)
<br><br>OpenFake는 다음과 같은 특징을 갖는 대규모 정치·사회 맥락 기반 딥페이크 벤치마크입니다.

- 규모
    - 약 300만 장의 real 이미지 + 캡션
    - 여기에 대응하는 약 100만 장의 synthetic 이미지(최신 생성 모델들로 생성)
- 목적
    - 기존 GAN/얼굴 교체 위주 벤치마크의 한계를 넘어,
        - 군중, 시위, 재난, 조작된 표지판, 합성 스크린샷 등
정치·사회적 맥락이 강한 실제 딥페이크 시나리오를 다룸.

- 구성
    - Real 이미지
        - 웹 대규모 이미지–텍스트 데이터셋 LAION-400M에서 추출한 정치·사회 관련 이미지 사용.
        - Qwen2.5-VL 기반 필터링으로
            - 실제 인물 얼굴이 포함되거나
            - 정치적으로/뉴스 가치가 높은 사건인 경우만 선택.
    - Synthetic 이미지
        - Stable Diffusion, Flux, Midjourney, DALL·E 3, Imagen, GPT Image 등
여러 최신 diffusion·transformer 기반 생성 모델을 사용해 생성.
        - 해상도는 약 1메가픽셀 수준, 다양한 비율(9:16, 1:1, 16:9 등)을 포함.


OpenFake 데이터셋은 Hugging Face 상에서 CC BY-SA 4.0 라이선스로 공개되어 있으며,
일부 상용 생성 모델로 만든 하위 서브셋은 해당 모델의 “비경쟁(non-compete)” 조항에 따라
비상업적(non-commercial) 용도에 한해 사용할 수 있습니다.
> ⚠️ 본 리포지토리에서의 사용 역시 연구 및 비상업적 용도를 전제로 합니다.

## 3.2 LAION-400M (Upstream Real 이미지 출처)
- OpenFake의 real 이미지는 LAION-400M(CLIP-필터링 4억 쌍 이미지–텍스트 데이터셋)에서 가져온 것입니다.
- LAION-400M 메타데이터는 CC BY 4.0 라이선스로 배포됩니다.
- 본 프로젝트는 OpenFake를 통해 제공되는 가공 데이터만 직접 사용하며,
LAION 원본을 별도로 다운로드 또는 재분배하지 않습니다.

# 4. 이 리포지토리에서 사용한 데이터 구성
## 4.1 학습/평가에 실제로 사용한 split
학습 코드는 Hugging Face datasets를 통해 OpenFake의 기본 split을 그대로 사용합니다.
- 학습용: split="train"
- 평가용: split="test"

추가의 비공개 데이터나, 대회 외부의 다른 딥페이크 데이터셋은 사용하지 않았습니다.
데이터 증강은 이미지 열화(blur/노이즈/JPEG) 수준의 전처리로 제한됩니다
(자세한 내용은 아래 4.2 참고).

## 4.2 데이터 전처리 및 증강 방식
train.py 내부에서 다음과 같이 전처리가 이루어집니다.
### 1. 이미지 로딩
- Hugging Face datasets에서 불러온 image 필드를
    - PIL.Image 또는 numpy array → RGB 3채널 PIL 이미지로 통일.
### 2. 라벨 매핑
- 문자열 또는 숫자 라벨을
    - 0 = real
    - 1 = fake<br>
    로 반환
### 3. real 이미지 통계 기반 fake 열화 (real_train_stats.npz)
- real_train_stats.npz 파일에는 real 이미지의 해상도(res_vals), Laplacian blur 값(blur_vals) 분포가 저장되어 있습니다.
- label == 1(fake)일 때만 degrade_image_to_match_laion5 함수를 통해
다음 연산을 확률적으로 적용합니다.
    - real 해상도 분포에 맞춰 가벼운 스케일 조정
    - real blur 분포와 맞춰주는 Gaussian blur
    - 소량의 Gaussian noise
    - 품질 70~95 범위의 JPEG 인코딩·디코딩
- 목적: 실제(real) 이미지가 가진 해상도/블러/압축 특성과 fake를 맞춰,
탐지기가 단순한 압축 노이즈 차이만으로 구별하지 못하도록 하는 것.
### 4. 모델 입력 변환
- Hugging Face AutoImageProcessor를 사용해
    - 리사이즈, 정규화, 텐서 변환을 수행하고
    - pixel_values 텐서를 생성합니다.
- 최종적으로 {"pixel_values": ..., "label": ...} 형태로 Trainer에 전달됩니다.

# 5. 학습 코드 실행 방법
## 5.1 Conda 환경 생성 (deep_env.yaml)
### 1. 환경 생성
```bash
# 리포지토리 루트에서
cd baselines/SwinV2

# Conda 환경 생성
conda env create -f deep_env.yaml

# deep_env.yaml 안 name 필드에 맞게 활성화 (예: deep)
conda activate deep
```
### 2. 필수 요소
- Python 3.11
- PyTorch + CUDA 12.x (GPU 학습용)
- transformers, datasets, scikit-learn, opencv-python 등
주요 라이브러리는 deep_env.yaml에 명시됨.

GPU는 NVIDIA 1장, 24GB급 메모리 이상을 권장합니다
(배치 크기 32 기준, 여유가 적으면 --batch_size를 줄여도 됩니다).

## 5.2 일반 서버 / 워크스테이션에서 학습 실행
```bash
cd baselines/SwinV2
conda activate deep

python train.py \
  --output_dir ./swinv2-finetuned-openfake \
  --num_epochs 4 \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --num_workers 4 \
  --cache_dir .cache
```
- --output_dir : 체크포인트와 로그가 저장될 디렉토리
- --cache_dir : OpenFake 및 기본 모델 캐시 경로
(처음 실행 시 OpenFake 전체/부분을 내려받기 때문에 디스크 용량이 많이 필요합니다.
Hugging Face 데이터셋 카드 기준 전체 크기는 약 1TB 규모입니다. 
Hugging Face
)
## 5.3 (선택) Hugging Face 관련 환경변수
필수는 아니지만, 네트워크 안정성과 캐시 관리를 위해 다음 설정을 추천합니다.
```bash
export HF_HOME=.cache/hf
export HF_DATASETS_CACHE=.cache/hf/datasets

export HF_HUB_READ_TIMEOUT=180
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
```
# 6. 학습 코드 및 하이퍼파라미터 요약
- Optimizer 및 스케줄링: Hugging Face Trainer 기본값 사용
- 손실 함수: Cross-entropy (2-class)
- 평가 지표:
    - Accuracy
    - Precision / Recall / F1 (binary)
    - AUC-ROC
- 기타:
    - WANDB_PROJECT = "SwinOpenFake" 환경변수가 설정되어 있으면 Weights & Biases에 학습 로그를 기록하도록 구성되어 있습니다.
# 7. 라이선스 및 출처
## 7.1 데이터셋
- OpenFake 데이터셋
    - 제공: Complex Data Lab
    - 라이선스: CC BY-SA 4.0 (일부 상용 생성 모델 기반 서브셋은 비상업적 제한)
- Upstream: LAION-400M
    - 역할: OpenFake real 이미지의 원천 데이터셋
    - 라이선스: 메타데이터는 CC BY 4.0
## 7.2 모델
- 기본 SwinV2 Base 모델
    - 이름: microsoft/swinv2-base-patch4-window16-256
    - 라이선스: Apache-2.0
- 본 리포지토리의 파인튜닝 모델
    - 위 SwinV2 Base를 기반으로 OpenFake에 대해 파인튜닝한 모델이며,
Apache-2.0 + OpenFake(CC BY-SA 4.0 및 비상업 조건)을 모두 존중해야 합니다.
    - 즉, 연구·비상업적 용도의 딥페이크 탐지 연구 재현을 목적으로 사용되는 것을 가정합니다.

## 7.3 참고 링크
```text
OpenFake 데이터셋 (Hugging Face):
https://huggingface.co/datasets/ComplexDataLab/OpenFake

OpenFake 논문 (arXiv):
https://arxiv.org/abs/2509.09495

LAION-400M 소개:
https://laion.ai/laion-400-open-dataset/

SwinV2 Base 모델 카드 (Hugging Face):
https://huggingface.co/microsoft/swinv2-base-patch4-window16-256
```
# 8. 기타 특이사항 / 제한사항
- OpenFake 전체를 처음 다운로드할 경우, 네트워크 트래픽과 디스크 사용량이 매우 크며
(수백 GB 이상), 검증 환경에서는 부분 다운로드 또는 캐싱된 사본을 활용하는 것이 현실적입니다.
- 코드 구조상, 하이퍼파라미터(에폭, 배치 크기 등)를 줄여도
모델/데이터 파이프라인과 전처리·열화 로직은 동일하게 검증됩니다.
- 본 코드는 이미지 기반 딥페이크(합성 이미지) 탐지에 초점을 맞추고 있으며,
동영상 프레임 레벨 탐지나 얼굴 정합(face alignment) 등은 포함하지 않습니다.
