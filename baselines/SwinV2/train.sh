#!/usr/bin/bash
#
# VerifyAI - SwinV2 OpenFake Training (Aurora GPU Cluster)
#
# 이 스크립트는 클러스터에서 SwinV2 OpenFake 학습을 실행하기 위한 SLURM 스크립트입니다.
# 주요 기능:
#   - GPU/CPU/메모리 자원 요청
#   - 각 잡(job)마다 개별 SCRATCH 디렉토리 생성
#   - NFS 기반 영구 캐시(PERSIST_CACHE) <-> 잡 로컬 캐시(HF_HOME) 동기화
#   - 마지막 체크포인트 자동 탐색 후 재시작 (있을 경우)
#

#SBATCH -J swin_base                    # Job name
#SBATCH --gres=gpu:1                    # GPU 1개 사용
#SBATCH --cpus-per-gpu=8                # GPU당 CPU 코어 8개
#SBATCH --mem-per-gpu=29G               # GPU당 메모리 29GB
#SBATCH -p batch_ugrad                  # 사용할 partition (클러스터 환경에 맞게 변경 가능)
#SBATCH -w aurora-g1                    # 특정 노드 고정 (필요 시 제거 가능)
#SBATCH -t 1-0                          # 최대 실행 시간: 1일 (D-HH:MM 형식)
#SBATCH -o ../logs/swin_base_slurm-%A.out  # 로그 파일 경로 (%A = SLURM job ID)

# set -euo pipefail
# -e : 명령어가 실패하면 스크립트 즉시 종료
# -u : 설정되지 않은 변수를 사용하면 오류 처리
# -o pipefail : 파이프라인 중간에서 실패해도 실패로 인식
set -euo pipefail

echo "현재 경로:" "$(pwd)"
echo "사용 중인 Python:" "$(which python)"
echo "호스트명:" "$(hostname)"
date

# ---------- 환경 변수 설정 ----------
# Python 출력 버퍼링 비활성화 (실시간 로그 확인 용이)
export PYTHONUNBUFFERED=1

# Hugging Face Hub 네트워크 관련 설정
export HF_HUB_READ_TIMEOUT=180
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1  # 대용량 다운로드 가속 (hf-transfer 사용)

# ---------- 캐시 경로 설정 ----------

# (1) 영구 캐시(NFS 기반)
# - 여러 잡에서 공유할 수 있는 HF 캐시 경로
PERSIST_CACHE="${PWD}/.cache/hf"
mkdir -p "${PERSIST_CACHE}"

# (2) 잡별 SCRATCH 디렉토리 설정
# - 실제 학습 중에는 SCRATCH를 사용해 I/O 성능 확보
export SCRATCH_ROOT="/data/juventa23/scratch"   # 사용자별 scratch root
mkdir -p "${SCRATCH_ROOT}"
export SCRATCH="${SCRATCH_ROOT}/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH}"

# Hugging Face 전역 홈(모델/데이터/트랜스포머 캐시 전부 여기로)
export HF_HOME="$SCRATCH/.cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_DATASETS_CACHE"

echo "[df -h] SCRATCH 용량 체크:"
df -h "$SCRATCH" || true

# ---------- NFS 캐시 → SCRATCH로 시드 ----------
# 이미 PERSIST_CACHE에 저장된 HF 캐시가 있다면,
# 이를 SCRATCH(HF_HOME)로 rsync 하여 cold start 시간을 줄임.
if [ -d "$PERSIST_CACHE" ]; then
  echo "Seeding local cache from $PERSIST_CACHE -> $HF_HOME ..."
  rsync -a --info=progress2 \
    --exclude 'xet/**' \
    "$PERSIST_CACHE/" "$HF_HOME/" || true
fi

# ---------- 잡 종료 시 SCRATCH → NFS로 백업 ----------
sync_back() {
  echo "Syncing back local cache to $PERSIST_CACHE ..."
  mkdir -p "$PERSIST_CACHE"
  rsync -a --info=stats2 \
    --exclude 'xet/**' \
    "$HF_HOME/" "$PERSIST_CACHE/" || true
}
trap sync_back EXIT

# ---------- 파일 디스크립터 상한 조정 ----------
# 많은 파일을 동시에 여는 상황을 대비해 ulimit 증가 (실패해도 무시)
ulimit -n 4096 || true

# ---------- 체크포인트 자동 탐색 ----------
CKPT_DIR="${PWD}/swinv2-finetuned-openfake"

# checkpoint-* 디렉토리 중 가장 마지막(숫자 기준)을 선택
LAST_CKPT=$(ls -1d "${CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)
echo "Latest checkpoint: ${LAST_CKPT:-<none found>}"

# ---------- 학습 실행 ----------
# - train_test.py는 SwinV2를 OpenFake에 대해 학습하는 메인 스크립트입니다.
# - LAST_CKPT가 존재하면 해당 체크포인트에서 학습을 이어서 시작합니다.
python train_test.py \
  --output_dir "$CKPT_DIR" \
  ${LAST_CKPT:+--resume_from_checkpoint "$LAST_CKPT"} \
  --num_epochs 4 \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --num_workers 4 \
  --cache_dir "$HF_DATASETS_CACHE"

date
exit 0
