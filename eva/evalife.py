import json
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
import os

# 加载评分系统
sari = load("He-Xingwei/sari_metric")
rouge = load('rouge')
# 加载预训练模型和tokenizer
model_name = "He-Xingwei/LIFE-Corrector-GE"  # 替换为你的模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def load_data(filename):
    """从文件加载数据"""
    sources = []
    predictions = []
    references = []
    with open(filename, 'r') as fr:
        for line in fr:
            data_instance = json.loads(line)
            sources.append(data_instance['src'])
            references.append([data_instance['tgt']])
            # 将tgt和evidence输入模型生成预测
            input_text = data_instance['tgt'] + " " + data_instance['evidence']
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            output = model.generate(**inputs)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            predictions.append(prediction)
    return sources, predictions, references


def evaluate_and_save_results(sources, predictions, references, input_file, output_file):
    """使用SARI评分系统计算结果并保存到文件"""
    results = sari.compute(sources=sources, predictions=predictions, references=references)
    print(results)
    results2 = rouge.compute(predictions=predictions, references=references)
    print(results2)
    # 确保文件存在，并写入标题
    if not os.path.exists(output_file):
        with open(output_file, 'a') as fw:
            fw.write("Src, Tgt, Prediction, SARI\n")

    with open(output_file, 'a') as fw:
        for i in range(len(sources)):
            output = f"{results['keep']:.2f}, {results['del']:.2f}, {results['add']:.2f}, {results['sari']:.2f}, {100*results2['rouge2']:.2f}, {args.input_file}\n"
            fw.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factual Error Correction.")
    parser.add_argument('--input_file', type=str, default='beam_5.txt',
                        help='The input file for evaluation (a jsonlines).')
    parser.add_argument('--output_file', type=str,
                        help='The output file used to save the evaluation results.')

    args = parser.parse_args()

    # 加载数据
    sources, predictions, references = load_data(args.input_file)

    # 计算SARI评分并保存结果
    evaluate_and_save_results(sources, predictions, references, args.input_file, args.output_file)

    print(f"Evaluation completed. Results saved to {args.output_file}.")
