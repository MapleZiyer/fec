# -*- coding: UTF-8 -*-
# create input data for openai

import json
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create inputs for openai.')
    parser.add_argument('--correct', action='store_true', 
                        help='if True, use openai to correct the claim; otherwise, use openai to create data for facutal error correction.')  
    parser.add_argument('--use_evidence', action='store_true',
                    help='whether use evidences to revise the original claim.')
    parser.add_argument('--use_gold_evidence', action='store_true',
                        help='whether use gold or retrieved evidences.')
    parser.add_argument('--num_evidence', type=int, default=1,
                        help='the number of evidences used to revise the original claim.')
    parser.add_argument('--num_shot', type=int, default=8,
                    help='the number of demonstrated examples in the prompt.')
    parser.add_argument('--num_hops', type=int, default=2,
                        help='the number of hops.')


    args = parser.parse_args()
    print(args)
    # assert args.num_shot <=8
    verdict_list = ["REFUTES"]
    # mutation_list = ["negate", "substitute_similar", "substitute_dissimilar", "specific", "merge"]
    mutation_list = ["negate"]
    for mutation in mutation_list:
        dir_prefix = f'gold_{mutation}_{args.num_shot}-shot'
        if args.use_evidence:
            if args.use_gold_evidence:
                dir_prefix += f'_{args.num_evidence}-gold-evidence'
            else:
                dir_prefix += f'_{args.num_evidence}-retrieved-evidence'
        else:
            pass
        
        prompt_filename = f"../gold_prompts/refutes_{mutation}_{args.num_shot}_{args.num_hops}_evidence.txt"
        print(f"Load prompt from {prompt_filename}.")
        prompt = ""
        num_shot = 0
        with open(prompt_filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                if line.startswith('Original claim: '):
                    original_claim = line
                elif line.startswith('Mutated claim: '):
                    mutated_claim = line
                    num_shot +=1 
                    if args.correct:
                        prompt += mutated_claim
                        prompt += original_claim
                    else:
                        prompt += original_claim
                        prompt += mutated_claim
                    if num_shot == args.num_shot:
                        break
                else:
                    prompt += line
        assert num_shot == args.num_shot
        prompt += "\nEvidence: {evidence}\n"
        if args.correct:
            prompt += "Mutated claim: {{mutated_claim}}\n"
            prompt += "Original claim: "
        else:
            prompt += "Original claim: {original_claim}\n"
            prompt += "Mutated claim: "
        # print(prompt)
        
        for mode in ['dev','train']:
            filename = f'../seq2seq_data/{mode}.json'
            if args.correct:
                output_dir = "../correct_input_for_openai/"
                if mode in['dev', 'train']:
                    continue                
            else:
                output_dir = "../augment_input_for_openai/"
                if mode == 'test':
                    continue
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_filename = os.path.join(output_dir, dir_prefix + f'_{mode}.json')
            
            # load data 
            data_list = []
            num_words = 0
            count = 0
            with open(filename, 'r', encoding='utf-8') as fr:
                content = fr.read()
                datajson = json.loads(content)
                for instance in datajson:
                    if instance['label'] == "refutes":
                        continue
                    if args.use_evidence:
                        if args.use_gold_evidence:
                            collected_evidence = instance['evidence']
                        else:
                            collected_evidence = instance['evidence']
                        #collected_evidence = [f"{title}; {content}" for title, content in collected_evidence]
                        evidence = collected_evidence
                    else:
                        evidence = None

                    data_instance = {
                    "idx": count,
                    "tgt": instance['claim'],
                    "evidence": evidence,
                    }
                    count = count+1
                    data_list.append(data_instance)
            openai_inputs = []
            s = set()
            with open(output_filename, 'w', encoding='utf-8') as fw:
                for instance in data_list:
                    if args.correct:
                        input = prompt.format(evidence=instance['evidence'], mutated_claim=instance['src'])
                    else:
                        if instance['tgt'] not in s:
                            s.add(instance['tgt'])
                            input = prompt.format(evidence=instance['evidence'], original_claim=instance['tgt'])
                        else:
                            # if (instance['tgt'], instance['evidence']) in s:
                            #     print(instance['tgt'], instance['evidence'])
                            #     exit()
                            input = None
                    
                    if input is not None:
                        instance['input'] = input
                        openai_inputs.append(instance)
                        num_words += len(input.split())
                        # fw.write(json.dumps(instance)+'\n')
                json.dump(openai_inputs, fw, indent = 4)
            print(f"The output file is {output_filename} and has {num_words} words.")

                
                




