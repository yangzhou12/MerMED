#!/bin/bash
# Finetune + evaluate MerMED-FM across the downstream datasets, looping over
# train_sizes (few-shot fractions) x seeds x datasets. A free GPU is picked for
# each run. Edit the CONFIG block, then run:  bash scripts/finetune_mermed.sh

# ----------------------------- CONFIG (edit me) -----------------------------
# Released MerMED checkpoint (ViT-B/16).
MERMED_CKPT="../weights/MerMED.pth"
# Where per-run outputs/logs are written (folders: MerMED_<pct>_seed<seed>_{outputs,logs}).
output_path="/path/to/MerMED_Results"
# Dataset roots. Image paths inside each finetune_labels.csv are resolved against these.
MEDFM_ROOT="/path/to/MedFM"
ct_data_root="${MEDFM_ROOT}/ct"
pathology_data_root="${MEDFM_ROOT}/path"
ultrasound_data_root="${MEDFM_ROOT}/ultrasound"
cxr_data_root="${MEDFM_ROOT}/cxr/finetuning"
skin_data_root="${MEDFM_ROOT}/skin/finetuning"
# Eye finetuning images live separately (e.g. OpenEye_Source/Datasets/Finetuning).
eye_data_root="/path/to/eye/finetuning"

input_size=224
global_pool=avg
vit_model=vit_base_patch16

# Few-shot fractions (1.0 = full data) and random seeds.
declare -a train_sizes=(0.1 0.3 0.5 1.0)
declare -a seeds=(0 1 42 123 2025)

# Training hyperparameters (shared across runs).
declare -a training_params=(
    "--epochs 50"
    "--blr 5e-3"
    "--layer_decay 0.65"
    "--weight_decay 0.05"
    "--drop_path 0.2"
    "--use_amp"
)

# GPU monitoring thresholds (MiB) and poll interval (s).
INITIAL_MEMORY_THRESHOLD=500
USAGE_MEMORY_THRESHOLD=1000
CHECK_INTERVAL=10
# ---------------------------------------------------------------------------

# Maps a train_size to the percentage label used in output folder names.
declare -A train_size_map=(
    [0.1]=10  [0.2]=20  [0.3]=30  [0.4]=40
    [0.5]=50  [0.6]=60  [0.8]=80  [1.0]=100
)

# "<dataset_name> <num_classes> <data_root>"
declare -a datasets=(
    # ----- Eye (CFP) -----
    "APTOS2019 5 ${eye_data_root}"
    "CRFO-v4 7 ${eye_data_root}"
    "Glaucoma_fundus 3 ${eye_data_root}"
    "IDRiD 5 ${eye_data_root}"
    "JSIEC 39 ${eye_data_root}"
    "MESSIDOR2 5 ${eye_data_root}"
    "PAPILA 3 ${eye_data_root}"
    "DRCR_CFP 2 ${eye_data_root}"
    "FM-AMD 4 ${eye_data_root}"
    "FM-CKD 2 ${eye_data_root}"
    "FM-DR 5 ${eye_data_root}"
    "FM-Glaucoma 3 ${eye_data_root}"
    "FM-MMD 5 ${eye_data_root}"
    "Seed_Cataract 2 ${eye_data_root}"
    # ----- Eye (OCT) -----
    "OCTDL 7 ${eye_data_root}"
    "OCTID 5 ${eye_data_root}"
    "DRCR_OCT 2 ${eye_data_root}"
    # ----- Chest X-ray -----
    "COVIDx-CXR4 2 ${cxr_data_root}"
    "TBX11K 3 ${cxr_data_root}"
    "rsna_pneumonia 2 ${cxr_data_root}"
    "siim_acr_pneumothorax 2 ${cxr_data_root}"
    "CBIS_DDSM 3 ${cxr_data_root}"
    # ----- CT -----
    "chest-ctscan-images 4 ${ct_data_root}"
    "IQ-OTHNCCD 3 ${ct_data_root}"
    "SARS-COV-2 2 ${ct_data_root}"
    "HRCTCov19 2 ${ct_data_root}"
    "iCTCF 3 ${ct_data_root}"
    "RAPIER_CT 7 ${ct_data_root}"
    # ----- Pathology -----
    "CRC-VAL-HE-7K 9 ${pathology_data_root}"
    "PanNuke 19 ${pathology_data_root}"
    "Kather_Texture_2016 8 ${pathology_data_root}"
    "BreakHis 2 ${pathology_data_root}"
    "Chaoyang 4 ${pathology_data_root}"
    "LC25000 5 ${pathology_data_root}"
    "TCGA 32 ${pathology_data_root}"
    "RAPIER_Gastric 3 ${pathology_data_root}"
    "MIDOG25 2 ${pathology_data_root}"
    "AMi-Br 2 ${pathology_data_root}"
    # ----- Ultrasound -----
    "BUSC 2 ${ultrasound_data_root}"
    "BUSI 3 ${ultrasound_data_root}"
    "US3M 2 ${ultrasound_data_root}"
    "BrEaST 2 ${ultrasound_data_root}"
    # ----- Skin -----
    "BCN20000 9 ${skin_data_root}"
    "Derm7pt 2 ${skin_data_root}"
    "Dermnet 23 ${skin_data_root}"
    "HAM10000_clean 7 ${skin_data_root}"
    "pad-ufes 6 ${skin_data_root}"
    "HIBA 2 ${skin_data_root}"
    "MSKCC 2 ${skin_data_root}"
    "DDI 2 ${skin_data_root}"
)

get_gpu_memory_usage() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null
}

is_gpu_available() {
    local mem
    mem=$(get_gpu_memory_usage "$1")
    [[ "$mem" =~ ^[0-9]+$ ]] && [ "$mem" -lt "$INITIAL_MEMORY_THRESHOLD" ]
}

get_available_gpu() {
    local num_gpus
    num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        if is_gpu_available "$gpu_id"; then echo "$gpu_id"; return 0; fi
    done
    return 1
}

# Wait until a launched job actually occupies the GPU (or time out).
monitor_gpu_usage() {
    local gpu_id=$1 attempt=0 mem
    while [ $attempt -lt 30 ]; do
        sleep "$CHECK_INTERVAL"
        mem=$(get_gpu_memory_usage "$gpu_id")
        if [[ "$mem" =~ ^[0-9]+$ ]] && [ "$mem" -gt "$USAGE_MEMORY_THRESHOLD" ]; then
            echo "GPU $gpu_id in use (${mem} MiB)"; return 0
        fi
        ((attempt++))
    done
    echo "Warning: GPU $gpu_id never showed expected usage"; return 1
}

mkdir -p "${output_path}/logs"

for train_size in "${train_sizes[@]}"; do
    model_name="MerMED_${train_size_map[$train_size]}"
    for seed in "${seeds[@]}"; do
        for entry in "${datasets[@]}"; do
            dataset_name=$(echo "$entry" | cut -d ' ' -f 1)
            num_classes=$(echo "$entry" | cut -d ' ' -f 2)
            data_root=$(echo "$entry" | cut -d ' ' -f 3)

            checkpoint_dir="${output_path}/${model_name}_seed${seed}_outputs/"
            log_dir="${output_path}/${model_name}_seed${seed}_logs/"
            log_file="${dataset_name}_${model_name}_train${train_size}.log"

            cmd="python main_finetune.py \
                ${training_params[@]} \
                --batch_size 16 \
                --global_pool ${global_pool} \
                --model ${vit_model} \
                --input_size ${input_size} \
                --task ${dataset_name} \
                --nb_classes ${num_classes} \
                --data_path ${data_root}/${dataset_name} \
                --label_path ${data_root}/${dataset_name}/finetune_labels.csv \
                --train_size ${train_size} \
                --seed ${seed} \
                --output_dir ${checkpoint_dir} \
                --log_dir ${log_dir} \
                --finetune ${MERMED_CKPT}"

            # Wait for a free GPU, launch, and confirm it started.
            while true; do
                if gpu_id=$(get_available_gpu); then
                    echo "Launch ${dataset_name} (${model_name}, seed=${seed}) on GPU ${gpu_id}"
                    CUDA_VISIBLE_DEVICES=$gpu_id $cmd >> "${output_path}/logs/${log_file}" 2>&1 &
                    pid=$!
                    if monitor_gpu_usage "$gpu_id"; then break; fi
                    kill "$pid" 2>/dev/null; sleep 5
                fi
                echo "No free GPU, waiting..."; sleep 15
            done
        done
    done
done

echo "All MerMED finetuning jobs launched. See ${output_path}/logs/ for progress."
