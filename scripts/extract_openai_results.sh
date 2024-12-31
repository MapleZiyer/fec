
for input_filename in ../augment_output_for_openai/gold_negate_8-shot_2-retrieved-evidence_dev_gpt-3.5-turbo.jsonl \
                      ../augment_output_for_openai/gold_negate_8-shot_2-retrieved-evidence_train_gpt-3.5-turbo.jsonl 
do
    python ../models/extract_openai_results.py \
       --input_filename $input_filename
done

