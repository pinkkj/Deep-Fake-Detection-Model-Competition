# VerifyAI: OpenFake 기반 SwinV2 딥페이크 탐지

팀 **VerifyAI**

이 디렉터리는 SwinV2 Transformer를 **OpenFake** 데이터셋에 대해 파인튜닝하여  
딥페이크 이미지를 탐지하는 학습 스크립트를 포함합니다.  
학습 완료 후 OpenFake의 테스트 셋에 대한 평가까지 한 번에 수행합니다.


## 1. 디렉터리 구조

현재 디렉터리(`baselines/SwinV2`)는 다음과 같이 구성되어 있습니다.

```text
SwinV2/
 ├─ README.md          ← 이 설명 파일
 ├─ train.py           ← SwinV2 학습·평가 메인 스크립트
 ├─ train.sh           ← (선택) SLURM 클러스터용 학습 스크립트
 └─ deep_env.yaml      ← Conda 가상환경 설정 파일
```
> ✅(참고) ../real_train_stats.npz는 real 라벨 이미지들의 해상도 / Laplacian blur 분포를
미리 계산해 저장한 파일로, 학습 시 fake 이미지에 가벼운 열화(degradation)를 줄 때 사용됩니다

## 2. 환경 설정 (Environment)
### 2.1 Conda + deep_env.yaml 사용 (권장)
deep_env.yaml 파일을 이용하면, 대회에서 사용한 것과 거의 동일한 환경을 쉽게 만들 수 있습니다.
```bash
# 저장소 루트에서 시작했다고 가정
cd baselines/SwinV2

# Conda 환경 생성
conda env create -f deep_env.yaml

# deep_env.yaml 안의 name: 필드에 맞게 활성화 (예: deep)
conda activate deep
```
> ✅ 위 환경에는 Python 3.11, PyTorch CUDA 빌드, Hugging Face Transformers, datasets 등이
포함되어 있으며, train.py 실행에 필요한 주요 라이브러리가 모두 들어 있습니다.

### 2.2 (선택) pip로 직접 설치
만약 Conda 사용이 어려운 환경이라면, 동일한 버전이 아니더라도 deep_env.yaml 내 pip: 항목을 참고하여
필요 패키지를 수동으로 설치할 수 있습니다.

## 3. 데이터 (Dataset)
### 3.1 OpenFake (Hugging Face)
학습/평가는 Hugging Face Hub에 공개된 OpenFake 데이터셋을 사용합니다.
- 데이터셋 이름: ComplexDataLab/OpenFake
- 사용 split:
    - train: 학습용
    - test: 평가용

별도의 사전 다운로드는 필요 없습니다.
train.py 내부에서 load_dataset("ComplexDataLab/OpenFake", ...)를 호출하면,
Hugging Face Hub에서 자동으로 다운로드 및 캐시를 수행합니다.

### 3.2 Real 이미지 통계 (real_train_stats.npz)
real_train_stats.npz에는 real 라벨 이미지들로부터 미리 계산해 둔:
- blur_vals : Laplacian 기반 blur 값 분포
- res_vals : (height, width) 해상도 분포

가 저장되어 있으며, 학습 시 fake 이미지에 적용되는 다음과 같은 가벼운 열화에 사용됩니다.

- real 해상도 분포에 맞추는 가벼운 리사이즈
- Laplacian blur 분포를 참조한 Gaussian blur
- 소량의 Gaussian 노이즈 추가
- 낮은 품질의 JPEG 인코딩·디코딩

> ✅ 코드에서는 real_train_stats.npz를 현재 디렉터리(SwinV2) 기준 상대 경로
./real_train_stats.npz로 로드하도록 되어 있습니다. 파일 위치를 변경하지 않는 것을 권장합니다.

## 4. 학습 실행 (Training)
### 4.1 단일 GPU 서버 / 로컬 터미널에서 실행
GPU가 장착된 Linux 서버 또는 워크스테이션에서 다음과 같이 실행할 수 있습니다.
```bash
# 가상환경 활성화
conda activate deep   # 위 2.1에서 생성한 환경 이름

# SwinV2 디렉터리로 이동
cd baselines/SwinV2

# 학습 + 평가 실행
python train.py \
  --output_dir ./swinv2-finetuned-openfake \
  --num_epochs 4 \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --num_workers 4 \
  --cache_dir .cache
```
- --output_dir : 체크포인트 및 로그가 저장될 디렉터리
- --num_epochs : 학습 epoch 수 (기본 4)
- --batch_size : 배치 크기
- --learning_rate : 학습률
- --num_workers : 데이터로더 worker 수
- --cache_dir : Hugging Face 데이터셋 캐시 위치 (없으면 자동 생성)

학습이 끝나면, 스크립트 내부에서 OpenFake test split에 대한 평가를 수행하고
Accuracy, F1, AUC-ROC 등의 지표를 출력합니다.

### 4.2 (선택) Hugging Face 관련 환경변수
필수는 아니지만, 네트워크/캐시 관리에 도움이 되는 설정입니다.

```bash
# 캐시 위치 명시 (원하지 않으면 생략 가능)
export HF_HOME=.cache/hf
export HF_DATASETS_CACHE=.cache/hf/datasets

# 네트워크/로그 관련 옵션 (선택)
export HF_HUB_READ_TIMEOUT=180
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
```

이 변수가 없더라도 학습은 정상적으로 동작합니다.
단지 캐시 디렉터리 위치와 Hugging Face Hub 동작을 조금 더 명시적으로 제어하기 위한 용도입니다.

## 5. (선택) SLURM 클러스터에서 실행

이 디렉터리의 train.sh는 GPU 클러스터와 같은 SLURM 기반 환경에서
학습을 제출하기 위한 예시 스크립트입니다.
```bash
# Aurora 예시
sbatch train.sh
```
- train.sh의 주요 역할:
    - #SBATCH 옵션을 통해 GPU/CPU/메모리 자원을 요청
    - 잡(job)별 SCRATCH 디렉토리를 생성하여 Hugging Face 캐시를 로컬 디스크에 저장
    - 필요 시 NFS 기반 영구 캐시와 rsync를 통해 캐시를 재사용
    - swinv2-finetuned-openfake 폴더 내 checkpoint-* 디렉토리를 자동 탐색하여,
가장 최신 체크포인트에서 학습을 이어서 시작 (존재하는 경우)

> ✅ 일반적인 단일 서버/워크스테이션 환경에서 재현할 때는
이 스크립트를 사용하지 않고, 4.1의 python train.py 명령만 사용하면 충분합니다.

## 6. 재현 체크리스트
동일한 실험을 재현하기 위해 필요한 조건은 다음과 같습니다.
### 1. 환경
- Linux (x86_64) + NVIDIA GPU + CUDA 12.x 호환 드라이버
- Conda 또는 그에 준하는 가상환경 툴
- deep_env.yaml을 이용해 환경 생성 후 conda activate deep
### 2. 네트워크
- https://huggingface.co 접근 가능
(OpenFake 데이터셋 및 microsoft/swinv2-base-patch4-window16-256 모델 다운로드용)

### 3. 파일 구조
- 이 디렉터리 구조(baselines/SwinV2)를 그대로 유지
- real_train_stats.npz가 train.py와 동일한 위치에 존재

### 4. 실행
- 학습 및 평가:
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

위 조건이 충족되면, 제출된 코드와 동일한 설정으로
SwinV2 기반 OpenFake 딥페이크 탐지 모델을 재학습 및 평가할 수 있습니다.
