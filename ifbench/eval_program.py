import requests
import dspy
import requests
from sklearn.metrics import f1_score, recall_score
import re
import ujson as json
import random
from tqdm import tqdm
import csv
from collections import defaultdict
import math
import random
import sys
import ast
import time
from tqdm.asyncio import tqdm_asyncio
import asyncio
from Datasets.IFBench.evaluation_lib import InputExample
import Datasets.IFBench.evaluation_lib as evaluation_lib
from Datasets.IFBench.IFevalG_instruction import ifevalg_evaluation_lib
import nltk
nltk.download('punkt_tab')

from alpha_evolve_evaluator.evaluator import EvalResult


def load_jsonl_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # 按行分割内容
        lines = response.text.strip().split('\n')
        
        # 解析每一行的JSON
        data = []
        for i, line in enumerate(lines, 1):
            if line.strip():  # 跳过空行
                try:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"第 {i} 行JSON解析错误: {e}")
                    print(f"问题行内容: {line[:100]}...")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None


def cal_outputs(user_prompt, res, mode):
    data_InputExample = InputExample(key=user_prompt["key"],
                                            instruction_id_list=user_prompt["instruction_id_list"],
                                            prompt=user_prompt["prompt"],
                                            kwargs=[item for item in user_prompt["kwargs"]])
    response_data_dict = {}
    response_data_dict[user_prompt["prompt"]] = res
    
    if mode == "train" or mode == "valid":
        output = ifevalg_evaluation_lib.test_instruction_following_strict(data_InputExample, response_data_dict)
    elif mode == "test":
        output = evaluation_lib.test_instruction_following_strict(data_InputExample, response_data_dict)
    else:
        raise ValueError("mode must be one of: 'train', 'valid' and 'test'")
    return output


def build_feedback_prompt_no_llm(
        system_prompts, 
        majority_answers,
        group_answers,
        system_idx,
    ):
    header = (
        "When inference this program on some specific samples, we get the following feedbacks:\n"
    )
    questions_str = []
    for i in range(len(majority_answers)):
        majority = majority_answers[i]
        group = group_answers[i]

        descriptions = majority[system_idx]['description_list']
        follows = majority[system_idx]['follow_instruction_list']

        # 一一对应展示
        pairs_str = "\n".join(
            [f"- result {j}: (instruction: {d}, follow_instruction: {f})"
             for j, (d, f) in enumerate(zip(descriptions, follows))]
        )

        block = (
            f"Instruction {i}: {majority[system_idx]['prompt']}\n"
            f"answer: {majority[system_idx]['answer']}\n"
            f"final_answer: {majority[system_idx]['final_answer']}\n"
            f"results:\n{pairs_str}\n"
        )
        questions_str.append(block)

    return header + "\n".join(questions_str)

def build_feedback_prompt_no_mv(
        system_prompts, 
        majority_answers,
        group_answers,
        system_idx,
    ):
    questions_str = []
    for i in range(len(majority_answers)):
        majority = majority_answers[i]

        descriptions = majority[system_idx]['description_list']
        follows = majority[system_idx]['follow_instruction_list']

        # 一一对应展示
        pairs_str = "\n".join(
            [f"- result {j}: (instruction: {d}, follow_instruction: {f})"
             for j, (d, f) in enumerate(zip(descriptions, follows))]
        )

        block = (
            f"Question {i}: {majority[system_idx]['prompt']}\n"
            f"answer: {majority[system_idx]['answer']}\n"
            f"final_answer: {majority[system_idx]['final_answer']}\n"
            f"results:\n{pairs_str}\n"
        )
        questions_str.append(block)

    return "\n".join(questions_str)

def build_feedback_prompt(
        system_prompts, 
        majority_answers,
        group_answers,
        system_idx,
    ):
    header = (
        "The current system prompt works together with other system prompts, "
        "and their outputs are combined through majority voting to form the final result.\n"
        f"Your task is to analyze **only this system prompt**:{system_prompts[system_idx]}, without evaluating the other two.\n\n"
        "The goal of **This system prompt** is: Given an instruction, first answer it directly, and then further calibrate the answer to produce a final response that best meets the instruction's requirements."
        "Below are the final response and the results of the final answer whether meets the instruction's requirements."
        "Please analyze them one by one:\n\n"
    )
    questions_str = []
    for i in range(len(majority_answers)):
        majority = majority_answers[i]
        group = group_answers[i]

        descriptions = majority[system_idx]['description_list']
        follows = majority[system_idx]['follow_instruction_list']

        # 一一对应展示
        pairs_str = "\n".join(
            [f"- result {j}: (instruction: {d}, follow_instruction: {f})"
             for j, (d, f) in enumerate(zip(descriptions, follows))]
        )

        block = (
            f"Instruction {i}: {majority[system_idx]['prompt']}\n"
            f"answer: {majority[system_idx]['answer']}\n"
            f"final_answer: {majority[system_idx]['final_answer']}\n"
            f"Final answer after majority voting: {group}\n"
            f"results:\n{pairs_str}\n"
        )
        questions_str.append(block)
    tail = (
        "\nCarefully reflect on the following:\n"
        "1. For each instruction, what specific weaknesses does this system prompt show? \n" 
        "2. Across all instruction, what common issues or patterns can you identify?\n"
        "3. How could this system prompt be revised or improved so that, when combined with others, it contributes more effectively to better final results?\n\n"
    )

    return header + "\n".join(questions_str)

class IFbench_instruction(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question->answer")
        self.correct = dspy.ChainOfThought("question,answer->final_answer")

    def forward(self, question):
        try:
            answer = self.prog(question=question).answer
            final_answer = self.correct(question=question,answer=answer).final_answer
        except:
            answer = "Warning: Repetition phenomenon occurred, output is incomplete."
            final_answer = "Warning: Repetition phenomenon occurred, output is incomplete."
        
        return {"question":question,"answer":answer,"final_answer":final_answer}
    
async def get_ifbench_response_per_sys_priompt(user_prompt, sys_prompt, mode):
    prompts = re.findall(r"<system_prompt_\d+>.*?</system_prompt_\d+>", sys_prompt, flags=re.DOTALL)

    if len(prompts) != 2:
        raise ValueError(f"Expected 2 system prompts, but found {prompts}")

    sys_prompt_1, sys_prompt_2 = prompts

    system = IFbench_instruction()
    system.prog.predict.signature.instructions = sys_prompt_1
    system.correct.predict.signature.instructions = sys_prompt_2

    # 如果 system 是同步的，用 asyncio.to_thread
    res = await asyncio.to_thread(system, user_prompt["prompt"])
    if res["answer"] == None:
        res["answer"] = "Warning: Repetition phenomenon occurred, output is incomplete."
    if res["final_answer"] == None:
        res["final_answer"] = "Warning: Repetition phenomenon occurred, output is incomplete."
    #计算各指令完成情况

    output = cal_outputs(user_prompt, res["final_answer"], mode)


    if mode == "train":
        res_s = {"prompt":res["question"],
                    "answer":res["answer"],
                    "final_answer":res["final_answer"],
                    "description_list":output.description_list,
                    "follow_all_instructions":output.follow_all_instructions,
                    "follow_instruction_list":output.follow_instruction_list
                    }
    else:
        res_s = {"prompt":res["question"],
                    "answer":res["answer"],
                    "final_answer":res["final_answer"],
                    "follow_all_instructions":output.follow_all_instructions,
                    "follow_instruction_list":output.follow_instruction_list
                    }
    return res_s

async def get_response_major_voting(i, sampled_user_prompts, system_prompts, mode, lm, majority_answers, group_answers):
    user_prompt = sampled_user_prompts[i]

    # 并发处理 system_prompts
    tasks = [get_ifbench_response_per_sys_priompt(user_prompt, sp, mode) for sp in system_prompts]
    res_s = await asyncio.gather(*tasks)
    majority_answers[i] = res_s
    if len(system_prompts) == 1:
        group_answers[i] = res_s[0]['final_answer']
    else:
        integrated_prompt = f"""
    You have a question and several LLM answers.
    Your task is to select the answer that appears most frequently (the majority answer).
    Only output the **index number** (just the number in LLM Answers, no extra words).

    Question:
    {user_prompt}

    LLM Answers:
    {" ".join(f"{i}. {d['final_answer']}" for i, d in enumerate(res_s))}
    """
        # lm 同步，用 asyncio.to_thread
        group_index = await asyncio.to_thread(lm, integrated_prompt)
        if group_index[0] == None:
            group_index = "Warning: Repetition phenomenon occurred, output is incomplete."

        group_index = group_index[0].strip()   # 去掉前后空格换行

        # 提取第一个纯数字
        match = re.search(r"\d+", group_index)
        if match and int(match.group()) in range(len(res_s)):
            group_index = int(match.group())
        else:
            # fallback: 选择最短的答案
            lengths = [len(d['final_answer']) for d in res_s]
            group_index = min(range(len(res_s)), key=lambda i: lengths[i])

        group_answer = res_s[group_index]['final_answer']
        group_answers[i] = group_answer
        # print("single:", " ".join(f"{i}. {d['answer']}" for i, d in enumerate(res_s)))
        # print("group:", group_answer)
        # answers = [d["final_answer"] for d in res_s]
        # n = len(answers)

        # avg_f1s = []
        # for j, ans_j in enumerate(answers):
        #     scores = []
        #     for k, ans_k in enumerate(answers):
        #         if j == k:
        #             continue
        #         scores.append(string_pair_metrics(ans_j, ans_k)['f1'])
        #     avg_f1s.append(sum(scores) / len(scores) if scores else 0.0)

        # # 选择平均 F1 最高的答案
        # best_idx = max(range(n), key=lambda m: (avg_f1s[m], -len(answers[m])))
        # group_answer = answers[best_idx]
        # group_answers[i] = group_answer
async def process_feedback(i, system_prompts, majority_answers, group_answers, lm, feedbacks):
    if len(system_prompts) == 1:
        feedback_prompt = build_feedback_prompt_no_mv(
            system_prompts=system_prompts,
            majority_answers=majority_answers,
            group_answers=group_answers,
            system_idx=i
        )
    else:
        feedback_prompt = build_feedback_prompt(
            system_prompts=system_prompts,
            majority_answers=majority_answers,
            group_answers=group_answers,
            system_idx=i
        )
    feedback = await asyncio.to_thread(lm, feedback_prompt)
    feedbacks[i] = feedback[0]

async def release_one_group_for_questions(
        model_name = "openai/Qwen/Qwen3-8B",
        api_base=None,
        api_key=None,
        system_prompts=None,
        n_question_sample=10,
        max_length=1024,
        temperature=1.0,
        data_json_path=None,
        mode="test"
    ):

    requests.adapters.DEFAULT_TIMEOUT = 30
    lm = dspy.LM(model_name, api_key=api_key, api_base=api_base,max_tokens=max_length,temperature=temperature)
    dspy.settings.configure(
        lm=lm,)

    #读取数据
    questions_data = load_jsonl_from_url(data_json_path)
    #采样问题
    if n_question_sample == "full":
        sampled_questions_data = questions_data
    else:
        sampled_questions_data = random.sample(questions_data,n_question_sample)
    
    sampled_user_prompts = [item["prompt"] for item in sampled_questions_data]
    #sampled_instruct_id_list = [item["instruction_id_list"] for item in sampled_questions_data]

    #sampled_answers = [item["answer"] for item in sampled_questions_data] 
    
    majority_answers = [None] * len(sampled_questions_data)
    group_answers = [None] * len(sampled_questions_data)

    tasks = [
        get_response_major_voting(
            i, sampled_questions_data,
            system_prompts, mode, lm,
            majority_answers, group_answers
        )
        for i in range(len(sampled_questions_data))
    ]

    # 并发执行任务，并显示进度
    for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        await f
    
    #计算group的表现
    if n_question_sample == "full":
        scores = []
        for j in range(len(system_prompts)):
            group_follow_all_instructions = []
            for i in range(len(sampled_questions_data)):
                output = cal_outputs(sampled_questions_data[i], majority_answers[i][j]["final_answer"], mode)
                group_follow_all_instructions.append(output.follow_all_instructions)
            group_acc = sum(group_follow_all_instructions)/len(group_follow_all_instructions)
            scores.append({"acc": group_acc})
        group_follow_all_instructions = []
        for i in range(len(sampled_questions_data)):

            output = cal_outputs(sampled_questions_data[i], group_answers[i], mode)

            group_follow_all_instructions.append(output.follow_all_instructions)
        group_acc = sum(group_follow_all_instructions)/len(group_follow_all_instructions)

        all_final_answers = [majority_answers[i][0]["final_answer"] for i in range(len(sampled_questions_data))]

        return {"acc": group_acc}, all_final_answers
    else:
        # feedbacks = [None] * len(system_prompts)
        # feedback_tasks = [
        #     process_feedback(i, system_prompts, majority_answers, group_answers, lm, feedbacks)
        #     for i in range(len(system_prompts))
        # ]
        # # 并发执行，并用 tqdm 显示进度
        # for f in tqdm_asyncio.as_completed(feedback_tasks, total=len(feedback_tasks)):
        #     await f
        # return feedbacks
        feedbacks = [build_feedback_prompt_no_llm(
            system_prompts=system_prompts,
            majority_answers=majority_answers,
            group_answers=group_answers,
            system_idx=i
        )for i in range(len(system_prompts))]
        return feedbacks


async def evaluate(program, config=None) -> EvalResult:
    model_name = config.get("model_name", "openai/fallback/gpt-4.1-mini")
    api_base = config.get("api_base", "https://llm-api.p1.cn")
    api_key = config.get("api_key", "")
    n_question_sample = config.get("n_question_sample", 3)
    max_length = config.get("max_length", 4096)
    temperature = config.get("temperature", 1.0)
    train_json_path = config.get("train_json_path", "http://cloud.staging.p1.cn/v2/ai-raw/614f440a-f1a3-4142-829f-3947159ee272.jsonl")
    valid_json_path = config.get("valid_json_path", "http://cloud.staging.p1.cn/v2/ai-raw/815376a3-7b02-4e4e-b6e6-3aa4b342440e.jsonl")
    #feedback
    if isinstance(program, str):
        program = [program]

    print("start get feedback")
    feedbacks = await release_one_group_for_questions(model_name = model_name,
                                                 api_base=api_base,
                                                 api_key=api_key,
                                                 system_prompts=program,
                                                 n_question_sample=n_question_sample,
                                                 max_length=max_length,
                                                 temperature=temperature,
                                                 data_json_path=train_json_path,
                                                 mode = "train")

    #feedback
    print("start get matric")
    metric, all_final_answers = await release_one_group_for_questions(model_name = model_name,
                                                 api_base=api_base,
                                                 api_key=api_key,
                                                 system_prompts=program,
                                                 n_question_sample="full",
                                                 max_length=max_length,
                                                 data_json_path=valid_json_path,
                                                 mode = "valid")

    eval_result = EvalResult()
    eval_result.metrics = metric
    eval_result.feedback = feedbacks[0] + "==^&*(split-part)==" + "==^&*(split-val)==".join(all_final_answers)

    return eval_result
