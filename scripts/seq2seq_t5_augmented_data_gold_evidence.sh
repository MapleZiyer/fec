

echo "Start to train the t5 with openai augmented data for factual error correction."

# t5 models do not support fp16 training.
export NCCL_TIMEOUT=3600

CUDA_VISIBLE_DEVICES=0,1

for DATA_PREFIX in  gold_negate_8-shot_2-retrieved-evidence 
do
    NUM_DATA_INSTANCE=11023
    # 1. Train
    CUDA_VISIBLE_DEVICES=0,1 \
    python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=9538  \
        ../models/seq2seq_baseline.py  \
        --train_file ../openai_augmented_data/${DATA_PREFIX}_train_gpt-3.5-turbo.jsonl \
        --initialization t5-base \
	--model_path ~/.cache/huggingface/hub/models--google-t5--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1 \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --lr 4e-5 \
        --logging_steps 50 \
        --save_steps 50 --max_steps 400 \
        --do_train --use_evidence --num_evidence 1 --use_gold_evidence  \
        --num_data_instance $NUM_DATA_INSTANCE \
        --output_dir ../openai_augmented_checkpoints/${DATA_PREFIX}_${NUM_DATA_INSTANCE}-data-instance_2-gold-evidence 


    # 2 eval
    CUDA_VISIBLE_DEVICES=0,1 \
    python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=9538  \
        ../models/seq2seq_baseline.py \
        --initialization t5-base \
        --use_evidence --num_evidence 1 --use_gold_evidence \
        --model_path ../openai_augmented_checkpoints/${DATA_PREFIX}_${NUM_DATA_INSTANCE}-data-instance_2-gold-evidence --resume \
        --do_eval

    # 3 predict
    for num_beams in 5
    do
        python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=9538  \
            ../models/seq2seq_baseline.py   \
            --initialization t5-base \
            --per_device_eval_batch_size 64 \
            --use_evidence --num_evidence 1 --use_gold_evidence  \
            --model_path ../openai_augmented_checkpoints/${DATA_PREFIX}_${NUM_DATA_INSTANCE}-data-instance_2-gold-evidence --resume \
            --num_beams $num_beams \
            --do_predict
    done
done

