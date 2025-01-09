import json
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
import os

# 加载预训练模型和tokenizer
model_name = "./local_model/models--He-Xingwei--LIFE-Corrector-GE/snapshots/09af74df7b7497de3516fb20adeb78c26ff93202"  # 替换为你的模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()


def load_data(input_filename, output_filename):
    with open(input_filename, 'r') as fr, open(output_filename, 'w') as fw:
        for line in fr:
            data_instance = json.loads(line)
            input_data = {
                "mutated": data_instance["tgt"],
                "gold_evidence": data_instance["evidence"],
                "origin": ""
            }
            input_str = json.dumps(input_data, ensure_ascii=False)
            print(input_str)
            # 提取 tgt 和 evidence 输入模型生成预测
            inputs = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=512, padding=True)
            output = model.generate(**inputs)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)

            # 构建输出数据
            output_data = {
                "src": data_instance["src"],
                "tgt": data_instance["tgt"],
                "generate": prediction
            }

            # 写入文件
            fw.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            print(output_data)
            print('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factual Error Correction.")
    parser.add_argument('--input_file', type=str, default='beam_5.txt',
                        help='The input file for evaluation (a jsonlines).')
    parser.add_argument('--output_file', type=str,
                        help='The output file used to save the evaluation results.')

    args = parser.parse_args()

    # 加载数据
    load_data(args.input_file, args.output_file)

