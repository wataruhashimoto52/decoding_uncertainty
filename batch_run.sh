#!/bin/bash -eu
#SBATCH --job-name=llm_uncertainty
#SBATCH --cpus-per-task=8
#SBATCH --output=output.%J.log
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --chdir=/work/wataru-ha/decoding_uncertainty

export BASE_DIR=/work/wataru-ha/decoding_uncertainty

export MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-chat-hf
export CD_MODEL_NAME_OR_PATH=TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T

# trivia_qa, xsum, wmt19, humaneval
export DATASET_NAME=trivia_qa

# original, beam_search, diverse_beam_search, contrastive_search, contrastive_decoding, fsd, fsd_vec, dola, sled
export DECODING_STRATEGY=original


export HF_TOKEN="xxxx"   # set your huggingface token here

cd $BASE_DIR

module load singularity


singularity run --nv /work/wataru-ha/llm-uncertainty.sif /usr/bin/python3.10 -m spacy download en_core_web_sm

cd transformers
singularity run --nv /work/wataru-ha/llm-uncertainty.sif /usr/bin/pip install -e .
cd ..

singularity run --nv /work/wataru-ha/llm-uncertainty.sif /usr/bin/python3.10 src/run.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--cd_amateur_model_name_or_path $CD_MODEL_NAME_OR_PATH \
--dataset_name $DATASET_NAME \
--ue_methods_group information \
--batch_size 1 \
--use_small 0 \
--num_beams 7 \
--num_beam_groups 3 \
--diversity_penalty 1.0 \
--penalty_alpha 0.6 \
--top_k 5 \
--contrastive_decoding_alpha 0.1 \
--contrastive_decoding_beta 0.5 \
--fsd_topk 5 \
--fsd_alpha 0.3 \
--fsd_vec_topk 5 \
--fsd_vec_alpha 0.3 \
--sled_layers high \
--sled_mature_layer 32 \
--sled_evolution_rate 0.1 \
--sled_evolution_scale 10 \
--sled_evolution_lower_bound -100 \
--dola_layers low \
--repetition_penalty 1.0 \
--decoding_strategy $DECODING_STRATEGY
