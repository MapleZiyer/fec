# -*- coding: UTF-8 -*-
# run with python version>3.6
# pip install openai
import openai
import argparse
import json
import os
import time
from tqdm import tqdm

api_key_dict= {'OpenaiAccount':"sk-NVz2LEoGeiJ0vMTkt4nwTHestJiEoRcjs8aplkkAEjBPULme"}
openai.api_base = "https://api.bianxie.ai/v1"

def openai_generate(input_prompt, args, try_time=1):
    if try_time>args.retry_time:
        print(f"{args.retry_time} requests failed. We will kill this script.")
        exit()
        return None
    try:
        if args.model in ["text-ada-001", "text-babbage-001", "text-curie-001", "davinci", "code-davinci-002", "text-davinci-003"]: 
            response = openai.Completion.create(
                            model=args.model,
                            prompt=input_prompt,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature # 0 denotes greedy decoding
                        )
        else:
            response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                        {"role": "user", "content": input_prompt}
                                        ],
                            max_tokens=args.max_tokens,
                            temperature=args.temperature # 0 denotes greedy decoding
                        )
    except openai.error.RateLimitError:
        print(f"RateLimitError. Retry your request in {args.wait_time} seconds.")
        time.sleep(args.wait_time)
        return openai_generate(input_prompt, args, try_time+1)
    except openai.error.APIError:
        print(f"APIError. Retry your request in {args.wait_time} seconds.")
        time.sleep(args.wait_time)
        return openai_generate(input_prompt, args, try_time+1)
    except openai.error.Timeout:
        print(f"Timeout. Retry your request in {args.wait_time} seconds.")
        time.sleep(args.wait_time)
        return openai_generate(input_prompt, args, try_time+1)
    except:
        print(f"Error. Retry your request in {args.wait_time} seconds.")
        time.sleep(args.wait_time)
        return openai_generate(input_prompt, args, try_time+1)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='openai.')
    parser.add_argument('--input_filename', type=str, required=True) 
    parser.add_argument('--output_dir', type=str, required=True) 
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo", choices=["text-ada-001", "text-babbage-001", "text-curie-001", "davinci", "code-davinci-002", "text-davinci-003", "gpt-3.5-turbo"])   
    parser.add_argument('--max_tokens', type=int, default=3000, help='The number of max tokens for the generated output.')
    parser.add_argument('--temperature', type=float, default=0)   
    parser.add_argument('--wait_time', type=int, default=10, 
                        help='Retry your request after a specified seconds.')  
    parser.add_argument('--retry_time', type=int, default=10, 
                        help='The maximum number of retry times.')  
    parser.add_argument('--max_request', type=int, default=-1, 
                        help='The maximum number of request.')  
    parser.add_argument('--api_key', type=str, required=True) 
    args = parser.parse_args()
    print(args)
    openai.api_key=api_key_dict[args.api_key]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_filename = os.path.basename(args.input_filename.replace('.json', f'_{args.model}.jsonl'))
    output_filename = os.path.join(args.output_dir, output_filename)
    print(output_filename)
    generated_set = set()
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                idx = json.loads(line)['idx']
                generated_set.add(idx)
    print(f"{len(generated_set)} requests have been successfully called.")
    with open(args.input_filename, 'r', encoding='utf-8') as fr, open(output_filename, 'a', encoding='utf-8') as fw:
        data_list  = json.load(fr)
        num_request = 0
        for data_instance in tqdm(data_list):
            # data_instance = json.loads(line)
            # print(data_instance)
            if data_instance['idx'] in generated_set:
                continue
            input_prompt = data_instance['input']
            response = openai_generate(input_prompt, args)
            data_instance['response'] = response
            fw.write(json.dumps(data_instance)+'\n')
            num_request +=1 
            if num_request>=args.max_request:
                exit()
