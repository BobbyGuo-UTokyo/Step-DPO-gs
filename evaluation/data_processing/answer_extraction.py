import re
import regex

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string


def _fix_tan(string):
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")

    # replace \\ with \
    # string = string.replace("\\\\", "\\")
    # string = string.replace("\\\\", "\\")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "%")
    string = string.replace("\%", "%")
    # string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    # string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # if len(string.split("=")) == 2:
    #     if len(string.split("=")[0]) <= 2:
    #         string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = _fix_tan(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string

def extract_boxed_answers(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers

def extract_program_output(pred_str):
    """
    extract output between the last ```output\n...\n```
    """
    if "```output" not in pred_str:
        return ""
    if '```output' in pred_str:
        pred_str = pred_str.split('```output')[-1]
    if '```' in pred_str:
        pred_str = pred_str.split('```')[0]
    output = pred_str.strip()
    return output

def extract_answer(pred_str, exhaust=False):
    pred = []

    # import pdb; pdb.set_trace()

    if 'final answer is $' in pred_str and '$. I hope' in pred_str:
        tmp = pred_str.split('final answer is $', 1)[1]
        pred = [tmp.split('$. I hope', 1)[0].strip()]
    elif 'boxed' in pred_str:
        pred = extract_boxed_answers(pred_str)

        # import pdb; pdb.set_trace()

    elif ('he answer is' in pred_str):
        pred = [pred_str.split('he answer is')[-1].strip()]
    else:
        program_output = extract_program_output(pred_str)
        if program_output != "":
            # fall back to program
            pred.append(program_output)
        else: # use the last number
            pattern = '-?\d*\.?\d+'
            ans = re.findall(pattern, pred_str.replace(",", ""))
            if(len(ans) >= 1):
                ans = ans[-1]
            else:
                ans = ''
            if ans:
                pred.append(ans)

    # multiple line
    _pred = []
    for ans in pred:
        # ans = ans.strip().split("\n")[0]
        ans = ans.strip()
        
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        ans = strip_string(ans)
        _pred.append(ans)
    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""

def extract_vllm_gt_answer(reasoning, task):
    # "Answer: C. Rabbit"
    if task in ("ai2d", "aokvqa", "gllava_qa", "mathvision", "sqa"):
        gt_answer = reasoning.split("Answer:")[-1].strip().split(".")[0].strip()
    # Answer: [Switzerland, Spain]
    elif task in ("chartqa", "docvqa", "infovqa", "sa_gllava_qa", "textvqa"):
        gt_answer = reasoning.split("Answer:")[-1].strip()
    else:
        raise ValueError(f"Task {task} is not supported")
    return gt_answer

def extract_vllm_gt_tag_answer(reasoning):
    # LLaVA-CoT-100k
    gt_answer = reasoning.split("<CONCLUSION>")[-1].split("</CONCLUSION>")[0].strip()
    return gt_answer

def extract_vllm_model_answer(reasoning, task):
    # Ignore answers not following format
    if "nswer:" not in reasoning:
        return []
    answer = reasoning.split("nswer:")[-1].strip()
    if task in ("ai2d", "aokvqa", "gllava_qa", "mathvision", "sqa"):
        if "." in answer:
            answer = answer.split(".")[0].strip()
    # Answer: [Switzerland, Spain]
    elif task in ("chartqa", "docvqa", "infovqa", "sa_gllava_qa", "textvqa"):
        pass
    else:
        raise ValueError(f"Task {task} is not supported")
    answer = answer.strip()
    answer = answer.rstrip(".")
    answer = answer.rstrip("/")
    return [answer]

def extract_vllm_model_tag_answer(reasoning, task):
    # Ignore answers not following format
    if "<CONCLUSION>" not in reasoning:
        return []
    answer = reasoning.split("<CONCLUSION>")[-1].split("</CONCLUSION>")[0].strip()
    # # Answer: B. Fish, extract letter
    # if task in ("ai2d", "aokvqa", "gllava_qa", "mathvision", "sqa"):
    #     if "." in answer:
    #         answer = answer.split(".")[0].strip()
    # # Answer: [Switzerland, Spain], keep original answer
    # elif task in ("chartqa", "docvqa", "infovqa", "sa_gllava_qa", "textvqa"):
    #     pass
    # else:
    #     raise ValueError(f"Task {task} is not supported")
    # answer = answer.strip()
    # answer = answer.rstrip(".")
    # answer = answer.rstrip("/")
    return [answer]

def extract_math_answer(question, reasoning, task):
    answer = []
    for ans in extract_answer(reasoning, exhaust=True):
        if 'separated by commas' in question and all(ch not in ans for ch in '()[]'):
            answer.extend([a.strip() for a in ans.split(",")])
        elif regex.search(r"\\text\{\s*and\s*\}", ans):
            answer.extend([a.strip() for a in regex.sub(r"\\text\{\s*and\s*\}", "[SEP]", ans).split("[SEP]")])
        else:
            answer.append(ans.strip())
    return answer

def extract_math_few_shot_cot_answer(question, reasoning, task):
    if 'Problem:' in reasoning:
        reasoning = reasoning.split("Problem:", 1)[0]
    return extract_math_answer(question, reasoning, task)

def extract_last_single_answer(question, reasoning, task):
    return extract_answer(reasoning, exhaust=False)

def extract_gsm_few_shot_cot_answer(question, reasoning, task):
    if 'Q: ' in reasoning:
        reasoning = reasoning.split("Q: ", 1)[0]
    pred = [s for s in regex.findall(r'-?\d+\.?\d*', reasoning)]
    if pred:
        return pred[-1]
    else:
        return "[invalid]"

def extract_agieval_gaokao_mathcloze_few_shot_cot_test(question, reasoning, task):
    if '问题 ' in reasoning:
        reasoning = reasoning.split("问题 ", 1)[0]
    if '答案是' in reasoning:
        ans = reasoning.split('答案是', 1)[1].strip()
        ans = ans.split("\n")[0].strip()
        ans = [ans.strip("$")]
    else:
        ans = ['placeholder']
    return ans

def extract_agieval_gaokao_mathqa_few_shot_cot_test(question, reasoning, task):
    if '问题 ' in reasoning:
        reasoning = reasoning.split("问题 ", 1)[0]
    if '答案是' in reasoning:
        ans = reasoning.split('答案是', 1)[1].strip()
        ans = ans.split("\n")[0].strip()
    else:
        ans = 'placeholder'
    return ans

def extract_sat_few_shot_answer(question, reasoning, task):
    if 'Problem:' in reasoning:
        reasoning = reasoning.split("Problem:", 1)[0]
    patt = regex.search(r"the final answer is \(?(?P<ans>[abcd])\)?", reasoning.lower())
    if patt is not None:
        return patt.group('ans').upper()
    return 'placeholder'

def extract_ocwcourses_few_shot_answer(question, reasoning, task):
    if 'Problem:' in reasoning:
        reasoning = reasoning.split("Problem:", 1)[0]
    patt = regex.search(r"final answer is (?P<ans>.*)\. I hope it is correct.", reasoning)
    if patt is None:
        pred = "[invalid]"
        print(f"DEBUG >>>\n{reasoning}", flush=True)
    else:
        pred = patt.group('ans')
    return pred

def extract_mmlu_stem(question, reasoning, task):
    if 'Problem:' in reasoning:
        reasoning = reasoning.split("Problem:", 1)[0]
    return extract_sat_few_shot_answer(question, reasoning, task)

def extract_minif2f_isabelle(question, reasoning, task):
    if 'Informal:' in reasoning:
        reasoning = reasoning.split("Informal:", 1)[0]
    return reasoning.strip()

def extract_cmath_few_shot_test(question, reasoning, task):
    if '问题：' in reasoning:
        reasoning = reasoning.split("问题：", 1)[0]
    if '答案是' in reasoning:
        ans = reasoning.split('答案是', 1)[1].strip()
        ans = ans.split("\n")[0]
        ans = ans.strip("：")
        ans = ans.strip("。")
        try:
            ans = [s for s in regex.findall(r'-?\d+\.?\d*', ans)][-1]
        except:
            print(f"DEBUG CMATH: {reasoning}", flush=True)
            ans = "[invalid]"
    else:
        ans = extract_last_single_answer(question, reasoning, task)
    return ans
