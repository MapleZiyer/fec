# # use openai for fatucal error correction and create inputs for it.
# python ../models/create_openai_input.py --use_evidence --use_gold_evidence --num_evidence 2 --correct --num_shot 8
# python ../models/create_openai_input.py --use_evidence --num_evidence 2 --correct --num_shot 8

# use openai for data augmentation and create inputs for it.
# python ../models/create_openai_input.py --use_evidence --use_gold_evidence --num_evidence 2 --num_shot 8
python ../models/create_openai_input.py --use_evidence --num_evidence 2 --num_shot 8

