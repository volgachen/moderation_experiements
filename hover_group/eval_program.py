import requests
import dspy
import re
from collections import Counter
import ujson as json
import random
import asyncio
import time
from tqdm.asyncio import tqdm_asyncio
from dataclasses import dataclass, field

from openai import AsyncOpenAI, BadRequestError
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_not_exception_type, retry_if_exception_type, before_sleep_log, wait_exponential

from alpha_evolve_evaluator.evaluator import EvalResult, BaseConfig
from protocols import logger


def load_json_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # 按行分割内容
        data = json.loads(response.text.strip())
        return data
        
    except requests.exceptions.RequestException as e:
        logger.info(f"请求错误: {e}")
        return None


def discrete_retrieval_eval(supporting_facts, pred, trace=None):
    gold_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [doc[0] for doc in supporting_facts],
        )
    )
    found_titles = set(
        map(
            dspy.evaluate.normalize_text,
            pred,
        )
    )
    return int(gold_titles.issubset(found_titles))

def build_feedback_prompt_no_mv(
        questions, 
        ground_truths,
        system_prompts, 
        majority_answers,
        group_answers, 
        system_idx,
    ):
    header = (
        f"Your task is to analyze **this system prompt**: {system_prompts[system_idx]}\n\n"
        "The goal of **this system prompt** is: given an input claim, perform two rounds of retrieval and summarization "
        "whether the **true_answer** are a subset of the **pred_answer**.\n\n"
        "Below are the results of applying this system prompt to multiple claims. "
        "Please analyze them one by one:\n\n"
    )
    questions_str = []
    for i in range(len(questions)):
        q = questions[i]
        majority = majority_answers[i]
        gt = ground_truths[i]

        block = (
            f"Claim {i}: {q}\n"
            f"This system prompt’s output: {majority[system_idx]}\n"
            f"true_answer: {[doc[0] for doc in gt]}\n"
        )
        questions_str.append(block)
    tail = (
        "\nCarefully reflect on the following:\n"
        "1. For each claim, what specific weaknesses does this system prompt show? \n" 
        "2. Across all claims, what common issues or patterns can you identify?\n"
        "3. How could this system prompt be revised or improved so that, it contributes more effectively to better final results?\n\n"
    )
    return header + "\n".join(questions_str) + tail

def build_feedback_prompt(
        questions, 
        ground_truths,
        system_prompts, 
        majority_answers,
        group_answers, 
        system_idx,
    ):
    header = (
        "The current system prompt works together with other system prompts, "
        "and their outputs are combined through majority voting to form the final result.\n"
        f"Your task is to analyze **only this system prompt**:{system_prompts[system_idx]}, without evaluating the other two.\n\n"
        "The goal of **this system prompt** is: given an input claim, perform two rounds of retrieval and summarization "
        "whether the **true_answer** are a subset of the **final pred_answer**.\n\n"""
        "Below are the results of this system prompt on multiple questions. "
        "Please analyze them one by one:\n\n"
    )

    questions_str = []
    for i in range(len(questions)):
        q = questions[i]
        majority = majority_answers[i]
        group = group_answers[i]
        gt = ground_truths[i]
        system_prompt = system_prompts[system_idx]
        other = [majority[j] for j in range(len(system_prompts)) if j != system_idx]

        block = (
            f"Claim {i}: {q}\n"
            f"This system prompt’s output: {majority[system_idx]}\n"
            f"Other system prompts' output: {other}\n"
            f"Final pred_answer after majority voting: {group}\n"
            f"true_answer: {gt}\n"
        )
        questions_str.append(block)

    tail = (
        "\nCarefully reflect on the following:\n"
        "1. For each question, what specific weaknesses does this system prompt show? \n" 
        "2. Across all questions, what common issues or patterns can you identify?\n"
        "3. How could this system prompt be revised or improved so that, when combined with others, it contributes more effectively to better final results?\n\n"
    )
    return header + "\n".join(questions_str) + tail

def safe_retrieve(retrieve_fn, query, retries=200, delay=0.2):
    for attempt in range(retries):
        if query == None:
            print(f"[Retrieve error], query is {query}")
            return []  # 最终失败返回空
        try:
            return retrieve_fn(query).passages
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay) # 随机等待，避免雪崩
            else:
                print(f"[Retrieve error] {e}, query {query}, attempt {attempt+1}/{retries}")
                return []  # 最终失败返回空

class HotpotQA_HoverMultiHop(dspy.Module):
    def __init__(self,top_k):
        super().__init__()
        self.k = top_k
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = safe_retrieve(self.retrieve_k, claim)
        hop1_docs_str = ' '.join([str(i+1)+". " + hop1_docs[i] for i in range(len(hop1_docs))])
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs_str
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = safe_retrieve(self.retrieve_k, hop2_query)
        hop2_docs_str = ' '.join([str(i+1)+". " + hop2_docs[i] for i in range(len(hop2_docs))])
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs_str
        ).summary

        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query

        hop3_docs = safe_retrieve(self.retrieve_k, hop3_query)
        # hop3_docs = ' '.join([str(i+1)+". " + hop3_docs[i] for i in range(len(hop3_docs))])

        hop_answer = dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
        answer = [c.split(" | ")[0] for c in hop_answer.retrieved_docs]
        return {"hop1_docs":hop1_docs,"summary_1":summary_1,"hop2_query":hop2_query,"hop2_docs":hop2_docs,"summary_2":summary_2,"hop3_query":hop3_query, "hop3_docs": hop3_docs, "answer": answer}



async def get_hotpotqa_response_per_sys_priompt(user_prompt, answer, sys_prompt, top_k_search, lm):
    prompts = re.findall(r"<system_prompt_\d+>.*?</system_prompt_\d+>", sys_prompt, flags=re.DOTALL)

    if len(prompts) != 4:
        raise ValueError(f"Expected 4 system prompts, but found {prompts}")

    sys_prompt_1, sys_prompt_2, sys_prompt_3, sys_prompt_4 = prompts
    
    system = HotpotQA_HoverMultiHop(top_k=top_k_search)

    system.summarize1.predict.signature.instructions = sys_prompt_1
    system.create_query_hop2.predict.signature.instructions = sys_prompt_2
    system.summarize2.predict.signature.instructions = sys_prompt_3
    system.create_query_hop3.predict.signature.instructions = sys_prompt_4

    # 如果 system 是同步的，用 asyncio.to_thread
    res = await asyncio.to_thread(system, user_prompt)
    perform = discrete_retrieval_eval(answer, res['answer'])
    res.update({"acc": perform})
    return res

def majority_elements(res_answers):
    n = len(res_answers)  # 子列表个数
    # 展平所有元素
    all_items = [elem for sublist in res_answers for elem in sublist]
    counter = Counter(all_items)

    # 保留出现次数 > n/2 的元素
    result = [elem for elem, cnt in counter.items() if cnt > n // 2]
    return result

async def get_response_major_voting(i, sampled_user_prompts, sampled_answers, system_prompts, top_k_search, lm, majority_answers, group_answers):
    user_prompt = sampled_user_prompts[i]
    answer = sampled_answers[i]

    # 并发处理 system_prompts
    tasks = [get_hotpotqa_response_per_sys_priompt(user_prompt, answer, sp, top_k_search, lm) for sp in system_prompts]
    res_s = await asyncio.gather(*tasks)
    majority_answers[i] = res_s

    if len(system_prompts) == 1:
        group_answers[i] = res_s[0]['answer']
    else:
        res_answers = [res_s[i]['answer'] for i in range(len(system_prompts))]
        group_answer = majority_elements(res_answers)
        group_answers[i] = group_answer

async def process_feedback(i, sampled_user_prompts, sampled_answers, system_prompts, majority_answers, group_answers, lm, feedbacks):
    if len(system_prompts) == 1:
        feedback_prompt = build_feedback_prompt_no_mv(
            questions=sampled_user_prompts,
            ground_truths=sampled_answers,
            system_prompts=system_prompts,
            majority_answers=majority_answers,
            group_answers=group_answers,
            system_idx=i
        )
    else:
        feedback_prompt = build_feedback_prompt(
            questions=sampled_user_prompts,
            ground_truths=sampled_answers,
            system_prompts=system_prompts,
            majority_answers=majority_answers,
            group_answers=group_answers,
            system_idx=i
        )
    feedback = await asyncio.to_thread(lm, feedback_prompt)
    feedbacks[i] = feedback[0]

async def release_one_group_for_questions(
    model_name="openai/Qwen/Qwen3-8B",
    wiki_url='http://20.102.90.50:2017/wiki17_abstracts',
    api_base=None,
    api_key=None,
    system_prompts=None,
    top_k_search=5,
    n_question_sample=10,
    max_length=1024,
    data_json_path=None,
    temperature=1.0
):
    lm = dspy.LM(model_name, api_key=api_key, api_base=api_base, temperature=temperature, max_tokens=max_length)

    dspy.settings.configure(
        lm=lm,
        rm=dspy.ColBERTv2(url=wiki_url),
    )

    # 读取数据
    questions_data = load_json_from_url(data_json_path)

    if n_question_sample == "full":
        sampled_questions_data = questions_data
    else:
        sampled_questions_data = random.sample(questions_data, n_question_sample)

    sampled_user_prompts = [item["claim"] for item in sampled_questions_data]
    sampled_answers = [item["supporting_facts"] for item in sampled_questions_data]

    majority_answers = [None] * len(sampled_user_prompts)
    group_answers = [None] * len(sampled_user_prompts)

    tasks = [
        get_response_major_voting(
            i, sampled_user_prompts, sampled_answers,
            system_prompts, top_k_search, lm,
            majority_answers, group_answers
        )
        for i in range(len(sampled_user_prompts))
    ]

    # 并发执行任务，并显示进度
    successed = 0
    for f in asyncio.as_completed(tasks):
        await f
        successed += 1
        logger.info(f"Successed: {successed}")

    if n_question_sample == "full":
        scores = []
        for j in range(len(system_prompts)):
            system_prompt_answer = [majority_answers[i][j]['answer'] for i in range(len(sampled_answers))]
            performs_group = [discrete_retrieval_eval(sampled_answers[i], system_prompt_answer[i]) for i in range(len(sampled_answers))]
            acc_mean_group = sum(performs_group) / len(performs_group)
            scores.append({"acc": acc_mean_group})

        performs_group = [discrete_retrieval_eval(sampled_answers[i], group_answers[i]) for i in range(len(sampled_answers))]
        acc_mean_group = sum(performs_group) / len(performs_group)
        return {"acc": acc_mean_group}, system_prompt_answer
    else:
        feedbacks = [None] * len(system_prompts)
        feedback_tasks = [
            process_feedback(i, sampled_user_prompts, sampled_answers, system_prompts, majority_answers, group_answers, lm, feedbacks)
            for i in range(len(system_prompts))
        ]
        # 并发执行，并用 tqdm 显示进度
        for f in tqdm_asyncio.as_completed(feedback_tasks, total=len(feedback_tasks)):
            await f
        return feedbacks, None


async def evaluate_single(program, config=None) -> EvalResult:
    parent_program = config["insights"][0]
    model_name = config.get("model_name", "openai/gpt-4.1-mini")
    wiki_url = config.get("wiki_url", 'http://20.102.90.50:2017/wiki17_abstracts')
    api_base = config.get("api_base", 'https://llm-api.p1.cn/v1')
    api_key = config.get("api_key", "")
    n_question_sample = config.get("n_question_sample", 3)
    top_k_search = config.get("top_k_search", 10)
    max_length = config.get("max_length", 4096)
    temperature = config.get("temperature", 1.0)
    train_json_path = config.get("train_json_path", "http://cloud.staging.p1.cn/v2/ai-raw/1fecdb0e-398e-46af-b69a-7e2a831acc53.json")
    valid_json_path = config.get("valid_json_path", "http://cloud.staging.p1.cn/v2/ai-raw/b415517b-9e32-42a6-b121-db583a2dcf8a.json")

    if isinstance(program, str):
        program = [program]

    #feedback
    logger.info("start get metric")
    metric, all_final_answers = await release_one_group_for_questions(
        model_name = model_name,
        wiki_url=wiki_url,
        api_base=api_base,
        api_key=api_key,
        system_prompts=program,
        top_k_search=top_k_search,
        n_question_sample="full",
        max_length=max_length,
        temperature=temperature,
        data_json_path=valid_json_path
    )

    eval_results = EvalResult()
    eval_results.metrics = {"acc": parent_program["metrics"]["acc"]}
    eval_results.feedback = "None" + "==^&*(split-part)==" + json.dumps(all_final_answers, ensure_ascii=False)

    return eval_results


async def evaluate_group(program, config) -> EvalResult:
    val_results = config.pop("val_results")
    programs = config.pop("programs")

    model_name = config.get("model_name", "openai/gpt-4.1-mini")
    wiki_url = config.get("wiki_url", 'http://20.102.90.50:2017/wiki17_abstracts')
    api_base = config.get("api_base", 'https://llm-api.p1.cn/v1')
    api_key = config.get("api_key", "")
    n_question_sample = config.get("n_question_sample", 3)
    top_k_search = config.get("top_k_search", 10)
    max_length = config.get("max_length", 4096)
    temperature = config.get("temperature", 1.0)
    train_json_path = config.get("train_json_path", "http://cloud.staging.p1.cn/v2/ai-raw/1fecdb0e-398e-46af-b69a-7e2a831acc53.json")
    valid_json_path = config.get("valid_json_path", "http://cloud.staging.p1.cn/v2/ai-raw/b415517b-9e32-42a6-b121-db583a2dcf8a.json")

    questions_data = load_json_from_url(valid_json_path)
    sampled_answers = [item["supporting_facts"] for item in questions_data]

    performs_group = []
    for i in range(len(val_results[0])):
        val_per_question = [val_results[j][i] for j in range(len(val_results))]
        final_answer = majority_elements(val_per_question)
        performs_group_i = discrete_retrieval_eval(sampled_answers[i], final_answer)
        performs_group.append(performs_group_i)
    acc_mean_group = sum(performs_group) / len(performs_group)
    final_metrics = {"acc": acc_mean_group}

    #feedback
    logger.info("start get feedback")
    feedbacks, _ = await release_one_group_for_questions(
        model_name = model_name,
        wiki_url=wiki_url,
        api_base=api_base,
        api_key=api_key,
        system_prompts=programs,
        top_k_search=top_k_search,
        n_question_sample=n_question_sample,
        max_length=max_length,
        temperature=temperature,
        data_json_path=train_json_path
    )

    eval_results = EvalResult()
    eval_results.metrics = final_metrics
    eval_results.feedback = "==^&*(split-feedback)==".join(feedbacks)

    return eval_results


async def evaluate(program, config):
    if config.pop("enable_feedback", False):
        res = await evaluate_group(program, config)
    else:
        res = await evaluate_single(program, config)
    return res
