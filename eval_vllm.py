import argparse
import json
import pdb
import jsonlines
import os
import PIL
import io
import base64

from evaluation.eval.eval_script import eval_math 
from evaluation.data_processing.answer_extraction import extract_vllm_gt_answer, extract_vllm_model_answer

from vllm import LLM, SamplingParams
import torch
import sys
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"
VALID_DATASETS = (
    "ai2d",
    "aokvqa",
    "chartqa",
    "docvqa",
    "infovqa",
    # "gllava-align", # open answer
    "sa_gllava_qa",
    "gllava_qa",
    "mathvision",
    "sqa",
    "textvqa"
)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    image_extension = os.path.basename(image_path).split(".")[-1]
    return base64.b64encode(data).decode("utf-8"), image_extension

invalid_outputs = []

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def test_vllm(model, data_path, image_root=None, remainder=0, n_groups=MAX_INT, batch_size=1, tensor_parallel_size=1, args=None):
    
    save_path = args.save_path
    vllm_ins = []
    vllm_answers = []
    attributes = []
    if args.prompt == 'alpaca':
        problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
    elif args.prompt == 'alpaca-cot-step':
        problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\nLet's think step by step.\nStep 1: "
        )
    elif args.prompt == 'alpaca-cot-prefix':
        problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\nLet's think step by step.\n{prefix}"
        )
    elif args.prompt == 'deepseek-math':
        problem_prompt = (
            "User: {instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:"
        )
    elif args.prompt == 'deepseek-math-step':
        problem_prompt = (
            "User: {instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant: Let's think step by step.\nStep 1: "
        )
    elif args.prompt == 'qwen2-boxed':
        problem_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    elif args.prompt == 'qwen2-boxed-step':
        problem_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\nLet's think step by step.\nStep 1: "
        )
    elif args.prompt == 'qwen2-vl-step':
        problem_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<image>\nAnswer the following question with information observed in the given image.\n"
            "{instruction}\n"
            # "If you are asked to give a short answer, just give a short answer as your final answer. "
            # "If you are asked to give a letter or a number, just give a letter or a number as your final answer. "
            # "Your answer must be strictly based on information from the image. If you cannot find the answer with given information, just give \"I don\'t know\" as your final answer. "
            "In your final answer, just include a short answer like a number, a letter, or a few words.\n"
            "Please reason step by step, and give your final answer after \"Answer: \".<|im_end|>\n"
            "<|im_start|>assistant\nLet's think step by step.\nStep 1: "
        )
    elif args.prompt == 'qwen2-boxed-prefix':
        problem_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\nLet's think step by step.\n{prefix}"
        )

    print('prompt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            # Filter dataset
            try:
                dataset_name = [name for name in VALID_DATASETS if name in item["id"]][0]
            except:
                continue
            # Get image path
            image_path = item["image"]
            # Filter single modal data
            if not isinstance(image_path, str):
                continue
            if image_root is not None:
                image_path = os.path.join(image_root, image_path)
            question = item["conversations"][0]["value"]
            item["answer"] = extract_vllm_gt_answer(item["conversations"][1]["value"], task=dataset_name)
            if "prefix" in item:
                temp_instr = problem_prompt.format(instruction=question, prefix=item['prefix'])
            else:
                temp_instr = problem_prompt.format(instruction=question)
            vllm_ins.append({"prompt": temp_instr, "image_path": image_path})
            temp_ans = item['answer']
            vllm_answers.append(temp_ans)
            attribute = {'task': dataset_name}
            if 'image' in item:
                attribute['image_path'] = image_path
            attributes.append(attribute)

    print("args.seed: ", args.seed)
    print('length ===', len(vllm_ins))
    vllm_ins = vllm_ins[remainder::n_groups]
    vllm_answers = vllm_answers[remainder::n_groups]
    attributes = attributes[remainder::n_groups]

    print("processed length ===", len(vllm_ins))
    vllm_ins = vllm_ins * args.rep
    vllm_answers = vllm_answers * args.rep
    attributes = attributes * args.rep

    print('total length ===', len(vllm_ins))
    batch_vllm_ins = batch_data(vllm_ins, batch_size=batch_size)

    sampling_params = SamplingParams(temperature=args.temp, top_p=args.top_p, max_tokens=2048)
    print('sampling =====', sampling_params)
    if not os.path.exists(save_path):
        llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, dtype=torch.bfloat16, seed=args.seed)

        # prompt = batch_vllm_ins[0][0]
        # base64_image, image_extention = encode_image(prompt["image_path"])
        # messages = [{
        #     "role": "user",
        #     "content": [
        #         {"type": "text", "text": "USER: <image>\nWhat is the content of this image?\nASSISTANT:"},
        #         {"type": "image_url", "image_url": {"url": f"data:image/{image_extention};base64,{base64_image}"}}
        #     ]
        # }]
        # completions = llm.chat(messages, sampling_params)
        # for output in completions:
        #     generated_text = output.outputs[0].text
        #     print(generated_text)
        # breakpoint()

        res_completions = []
        for idx, prompts in enumerate(batch_vllm_ins):
            assert isinstance(prompts, list)
            all_messages = []
            for prompt in prompts:
                base64_image, image_extention = encode_image(prompt["image_path"])
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_extention};base64,{base64_image}"}},
                        {"type": "text", "text": prompt["prompt"]}
                    ]
                }]
                all_messages.append(messages) # batch
            completions = llm.chat(all_messages, sampling_params=sampling_params, use_tqdm=True)
            for output in completions:
                generated_text = output.outputs[0].text
                res_completions.append(generated_text)
    else:
        res_completions = []
        with open(save_path) as f:
            items = json.load(f)
        for idx, item in enumerate(items):
            res_completions.append(item['completion'])

    to_save_list = []
    results = []
    for idx, (prompt, completion, prompt_answer, attribute) in enumerate(zip(vllm_ins, res_completions, vllm_answers, attributes)):

        if isinstance(prompt_answer, str) and prompt_answer.startswith("\\text{"):
            prompt_answer = remove_text(prompt_answer)

        if "The answer is:" in completion and (isinstance(prompt_answer, list) and len(prompt_answer) == 1 and "\\begin{pmatrix}" in prompt_answer[0]):
            prompt_answer[0] = prompt_answer[0].replace("\\\\", "\\")
            completion = completion.replace("\\\\", "\\")

        item = {
            'question': prompt,
            'model_output': completion,
            'image_path': attribute['image_path'],
            'prediction': extract_vllm_model_answer(completion, task=attribute["task"]),
            'answer': prompt_answer if isinstance(prompt_answer, list) else [prompt_answer],
        }

        if len(item['prediction']) == 0:
            invalid_outputs.append({'question': prompt, 'output': completion, 'answer': item['prediction']})
            res = False
            extract_ans = None
        else:
            extract_ans = item['prediction']
            res = eval_math(item)

        results.append(res)

        to_save_dict = {
            'prompt': prompt,
            'completion': completion,
            'extract_answer': extract_ans,
            'prompt_answer': prompt_answer,
            'result': res,
        }
        to_save_dict.update(attribute)
        to_save_list.append(to_save_dict)

    acc = sum(results) / len(results)
    # print('valid_outputs===', invalid_outputs)
    print('len invalid outputs ====', len(invalid_outputs))
    print('n_groups===', n_groups, ', remainder====', remainder)
    print('length====', len(results), ', acc====', acc)

    try:
        with open(save_path, "w+") as f:
            json.dump(to_save_list, f, indent=4)
    except:
        import pdb; pdb.set_trace()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--image_root", type=str, default=None)  # data path
    parser.add_argument("--remainder", type=int, default=0) # index
    parser.add_argument("--n_groups", type=int, default=1)  # group number
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--prompt", type=str, default='alpaca')
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rep", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_vllm(model=args.model, data_path=args.data_file, image_root=args.image_root, remainder=args.remainder, n_groups=args.n_groups, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size, args=args)
