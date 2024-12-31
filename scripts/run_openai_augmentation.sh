# use openai for data augmentation

for input_filename in ../augment_input_for_openai/gold_negate_8-shot_2-retrieved-evidence_train.json \
                      ../augment_input_for_openai/gold_negate_8-shot_2-retrieved-evidence_dev.json

do
    python ../models/run_openai.py \
        --max_request 200000 \
        --api_key OpenaiAccount \
        --output_dir ../augment_output_for_openai \
        --input_filename $input_filename --model gpt-3.5-turbo
done

