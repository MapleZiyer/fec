# -*- coding: UTF-8 -*-
import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='openai.')
    parser.add_argument('--input_filename', type=str, required=True) 
    parser.add_argument('--output_dir', type=str, default=None) 
    parser.add_argument('--correct', action='store_true', 
                    help='if True, use openai to correct the claim; otherwise, use openai to create data for facutal error correction.') 
    args = parser.parse_args()
    print(args)
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_filename)+ '/extracted'
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_filename = os.path.basename(args.input_filename)
    output_filename = os.path.join(args.output_dir, output_filename)
    print(output_filename)
    output_list = []
    with open(args.input_filename, 'r', encoding='utf-8') as fr, open(output_filename, 'w', encoding='utf-8') as fw:
        for line in fr:
            data_instance = json.loads(line)
            if "gpt-3.5-turbo.jsonl" in args.input_filename:
                generated_content = data_instance["response"]["choices"][0]["message"]["content"].strip()
            # elif "text-davinci-003.jsonl" in args.input_filename or "davinci.jsonl" in args.input_filename:
            else:
                generated_content = data_instance["response"]["choices"][0]["text"].strip()

            
            if "\n\nEvidence:" in generated_content:
                # assert "davinci.jsonl" in args.input_filename
                generated_content = generated_content.split("\n\nEvidence:")[0].strip()
            if args.correct:
                new_data_instance = {
                    "src": data_instance["src"],
                    "tgt": data_instance["tgt"],
                    "mutation_type": data_instance["mutation_type"],
                    "veracity": data_instance["veracity"],
                    "generated_text": generated_content
                    }
            else:
                new_data_instance = {
                    "mutated": generated_content,
                    "original": data_instance["tgt"],
                    "gold_evidence": data_instance["evidence"],
                    }
            output_list.append(new_data_instance)       
            fw.write(json.dumps(new_data_instance)+'\n')
        # json.dump(output_list, fw, indent=4)
