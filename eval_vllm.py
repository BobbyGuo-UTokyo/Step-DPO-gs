import argparse
import json
import pdb
import jsonlines
import os
import PIL
import io
import base64

from evaluation.eval.eval_script import eval_math 
from evaluation.data_processing.answer_extraction import extract_vllm_gt_answer, extract_vllm_gt_tag_answer, extract_vllm_model_answer, extract_vllm_model_tag_answer

from vllm import LLM, SamplingParams
import torch
import sys
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"
# VALID_DATASETS = (
#     "ai2d",
#     "aokvqa",
#     "chartqa",
#     "docvqa",
#     "infovqa",
#     # "gllava-align", # open answer
#     "sa_gllava_qa",
#     "gllava_qa",
#     "mathvision",
#     "sqa",
#     "textvqa"
# )

# For LLaVA-CoT-100k
VALID_DATASETS = (
    "coco",
    "ai2d",
    "vg",
    "geoqa+",
    "textvqa",
    "chartqa",
    "sqa",
    "docvqa",
    "gqa",
    "ocr_vqa",
    "CLEVR_v1.0",
    "web-landmark"
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
            "<|im_start|>user\n<image>\nYou are given a question to be answered together with information in the given image.\n"
            "{instruction}\n"
            "Your answer should strictly follow the four specific stages, question summary stage, image caption stage, reasoning steps stage, conclusion stage.\n"
            "In question summary stage, you must summarize what the problem is and what steps you may have to take to solve the problem, and the summary should be placed inside <SUMMARY> and </SUMMARY>.\n"
            "In image caption stage, describe the contents of the image, specifically focusing on details relevant to the question, and the caption should be placed inside <CAPTION> and </CAPTION>.\n"
            "In reasoning steps stage, outline a step-by-step thought process you would use to solve the problem based on the image. Please reason step by step, and place your reasoning steps inside <STEP_1> and </STEP_1>, <STEP_2> and </STEP_2>, and so on.\n"
            "In final conclusion stage, you must give a short direct answer like a number, a letter, or a few words. If it's a multiple choice question, the conclusion should only include the letter for the option without repeating what the option is. Place the final answer inside <CONCLUSION> and </CONCLUSION>.<|im_end|>\n"
            "<|im_start|>assistant\n<SUMMARY>"
            # "<|im_start|>user\nI have an image and a question that I want you to answer. I need you to strictly follow the format with four specific sections: SUMMARY, CAPTION, reasoning steps, and CONCLUSION. It is crucial that you adhere to this structure exactly as outlined and that the final answer in the CONCLUSION is brief and concise. To explain further: In SUMMARY, briefly explain what steps you'll take to solve the problem. In CAPTION, describe the contents of the image, specifically focusing on details relevant to the question. In reasoning steps, outline a step-by-step thought process you would use to solve the problem based on the image, and wrap in <Step_1> and </Step_1>, <Step_2> and </Step_2>, etc.. In CONCLUSION, give the final answer in a direct brief format. If it's a multiple choice question, the conclusion should only include the option without repeating what the option is. Here is an example on how the format should look like:\n<SUMMARY> [Summarize how you will approach the problem and explain the steps you will take to reach the answer.] </SUMMARY>\n<CAPTION> [Provide a detailed description of the image, particularly emphasizing the aspects related to the question.] </CAPTION>\n<Step_1> [Provide the first step of a chain-of-thought, logical explanation of the problem.] </Step_1>\n<Step_2> [Provide the second step of a chain-of-thought, logical explanation of the problem.] </Step_2>\n<CONCLUSION> [State the final answer in a clear and direct format. It must match the correct answer exactly.] </CONCLUSION> (Do not forget </CONCLUSION>!) Please apply this format meticulously to analyze the given image and answer the related question.<|im_end|>\n"
            # "<|im_start|>assistant\n<SUMMARY>"
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
                dataset_name = [name for name in VALID_DATASETS if name in item["image"]][0]
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
            # item["answer"] = extract_vllm_gt_answer(item["conversations"][1]["value"], task=dataset_name)
            item["answer"] = extract_vllm_gt_tag_answer(item["conversations"][1]["value"])
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
            try:
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
            except Exception as e:
                print(f"Error {e}, skip {idx}")
                for prompt in prompts:
                    res_completions.append("") # dummy invalid output
                continue
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
            'prediction': extract_vllm_model_tag_answer(completion, task=attribute["task"]),
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
