import requests
import dspy
import json
import re
import random
import asyncio
import time
from dataclasses import dataclass, field
from rouge_score import rouge_scorer

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


def normalize_text(text: str) -> set:
    # 标准化处理：小写化、去标点、分词
    text = re.sub(r'[^\w\s]', '', text.lower())
    return set(text.split())

def string_pair_metrics(pred_answer: str, true_answer: str) -> dict:
    pred_tokens = normalize_text(pred_answer)
    true_tokens = normalize_text(true_answer)
    
    # 计算TP/FP/FN
    tp = len(pred_tokens & true_tokens)
    fp = len(pred_tokens - true_tokens)
    fn = len(true_tokens - pred_tokens)
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"f1": f1, "recall": recall}

def build_feedback_prompt_no_llm(
        questions, 
        ground_truths,
        system_prompts, 
        majority_answers,
        group_answers, 
        system_idx,
    ):
    questions_str = []
    for i in range(len(questions)):
        q = questions[i]
        majority = majority_answers[i]
        gt = ground_truths[i]

        block = (
            f"Question {i}: {q}\n"
            f"This system prompt's output: {majority[system_idx]}\n"
            f"Ground truth answer: {gt}\n"
        )
        questions_str.append(block)
    return "\n".join(questions_str)
def build_feedback_prompt_no_mv(
        questions, 
        ground_truths,
        system_prompts, 
        majority_answers,
        group_answers, 
        system_idx,
    ):
    header = (
        f"Your task is to analyze **This system prompt**: {system_prompts[system_idx]}\n\n"
        "The goal of **This system prompt** is: given an input question, perform two rounds of retrieval and summary, such that the final generated answer must exactly match the true answer."
        "Below are the results of two rounds of retrieval and summary on multiple questions. "
        "Please analyze them one by one:\n\n"
    )
    questions_str = []
    for i in range(len(questions)):
        q = questions[i]
        majority = majority_answers[i]
        gt = ground_truths[i]

        block = (
            f"Question {i}: {q}\n"
            f"This system prompt's output: {majority[system_idx]}\n"
            f"True answer: {gt}\n"
        )
        questions_str.append(block)
    tail = (
        "\nCarefully reflect on the following:\n"
        "1. For each question, what specific weaknesses does this system prompt show? \n" 
        "2. Across all questions, what common issues or patterns can you identify?\n"
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
        "The goal of **This system prompt** is: given an input question, perform two rounds of retrieval and summary, such that the final generated answer must exactly match the true answer."
        "Below are the results of two rounds of retrieval and summary on multiple questions. "
        "Please analyze them one by one:\n\n"
    )

    questions_str = []
    for i in range(len(questions)):
        q = questions[i]
        majority = majority_answers[i]
        group = group_answers[i]
        gt = ground_truths[i]
        other = [majority[j] for j in range(len(system_prompts)) if j != system_idx]

        block = (
            f"Question {i}: {q}\n"
            f"This system prompt's output: {majority[system_idx]}\n"
            f"Other system prompts' output: {other}\n"
            f"Final result after majority voting: {group}\n"
            f"True answer: {gt}\n"
        )
        questions_str.append(block)

    tail = (
        "\nCarefully reflect on the following:\n"
        "1. For each question, what specific weaknesses does this system prompt show? \n" 
        "2. Across all questions, what common issues or patterns can you identify?\n"
        "3. How could this system prompt be revised or improved so that, when combined with others, it contributes more effectively to better final results?\n\n"
    )
    return header + "\n".join(questions_str) + tail

sem_wiki = asyncio.Semaphore(50)

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=5),
    stop=stop_after_attempt(200),
)
async def safe_retrieve(retrieve_fn, query, retries=200, delay=1):
    try:
        async with sem_wiki:
            res = retrieve_fn(query).passages
            await asyncio.sleep(3) # 随机等待，避免雪崩
            return res
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise e

sem_global = asyncio.Semaphore(80)

class HotpotQA_HoverMultiHop(dspy.Module):
    def __init__(self,top_k):
        super().__init__()
        self.k = top_k
        self.create_query_hop2 = dspy.ChainOfThought("question,summary_1->query")
        self.final_answer = dspy.ChainOfThought("question,summary_1,summary_2->answer")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("question,passages->summary")
        self.summarize2 = dspy.ChainOfThought("question,context,passages->summary")

    async def aforward(self, question):
      async with sem_global:
        # HOP 1
        hop1_docs = await safe_retrieve(self.retrieve_k, question)
        hop1_docs = ' '.join([str(i+1)+". " + hop1_docs[i] for i in range(len(hop1_docs))])
        try:
            summary_1 = await self.summarize1(
                question=question, passages=hop1_docs
            ).summary  # Summarize top k docs
        except:
            summary_1 = "Warning: Repetition phenomenon occurred, output is incomplete."

        # HOP 2
        try:
            hop2_query = await self.create_query_hop2(question=question, summary_1=summary_1).query
        except:
            hop2_query = "Warning: Repetition phenomenon occurred, output is incomplete."

        hop2_docs = await safe_retrieve(self.retrieve_k, hop2_query)
        hop2_docs = ' '.join([str(i+1)+". " + hop2_docs[i] for i in range(len(hop2_docs))])
        try:
            summary_2 = await self.summarize2(
                question=question, context=summary_1, passages=hop2_docs
            ).summary
        except:
            summary_2 = "Warning: Repetition phenomenon occurred, output is incomplete."

        try:
            answer = await self.final_answer(
                question=question, summary_1=summary_1, summary_2=summary_2
            ).answer
        except:
            answer = "Warning: Repetition phenomenon occurred, output is incomplete."


        return {"hop1_docs":hop1_docs,"summary_1":summary_1,"hop2_query":hop2_query,"hop2_docs":hop2_docs,"summary_2":summary_2,"answer":answer}



async def get_hotpotqa_response_per_sys_priompt(user_prompt, answer, sys_prompt, top_k_search):
    prompts = re.findall(r"<system_prompt_\d+>.*?</system_prompt_\d+>", sys_prompt, flags=re.DOTALL)

    if len(prompts) != 4:
        raise ValueError(f"Expected 4 system prompts, but found {prompts}, {sys_prompt}")

    sys_prompt_1, sys_prompt_2, sys_prompt_3, sys_prompt_4 = prompts
    
    system = HotpotQA_HoverMultiHop(top_k=top_k_search)

    system.summarize1.predict.signature.instructions = sys_prompt_1
    system.create_query_hop2.predict.signature.instructions = sys_prompt_2
    system.summarize2.predict.signature.instructions = sys_prompt_3
    system.final_answer.predict.signature.instructions = sys_prompt_4

    # 如果 system 是同步的，用 asyncio.to_thread
    res = await system.aforward(user_prompt)
    if res["answer"] == None:
        res["answer"] = "Warning: Repetition phenomenon occurred, output is incomplete."

    perform = string_pair_metrics(res["answer"], answer)
    res.update({"recall": perform["recall"], "f1": perform["f1"]})
    return res

async def get_response_major_voting(i, sampled_user_prompts, sampled_answers, system_prompts, top_k_search, lm, majority_answers, group_answers):
    user_prompt = sampled_user_prompts[i]
    answer = sampled_answers[i]

    # 并发处理 system_prompts
    tasks = [get_hotpotqa_response_per_sys_priompt(user_prompt, answer, sp, top_k_search) for sp in system_prompts]
    res_s = await asyncio.gather(*tasks)
    majority_answers[i] = res_s
    if len(system_prompts) == 1:
        group_answers[i] = res_s[0]['answer']
    else:
    #     integrated_prompt = f"""
    #  You have a question and several LLM answers. 
    #  Your task is to select the answer that appears most frequently (majority answer), and concise final answer in the following format:

    # **Final Answer:** <your answer here>

    # Question:
    # {user_prompt}

    # LLM Answers:
    # {" ".join(f"{i+1}. {d['answer']}" for i, d in enumerate(res_s))}
    # """
    #     # lm 同步，用 asyncio.to_thread

    #     group_answer = await asyncio.to_thread(lm, integrated_prompt)
    #     group_answer = group_answer[0]
    #     if group_answer == None:
    #         group_answer = "Warning: Repetition phenomenon occurred, output is incomplete."
    #     if "**Final Answer:**" in group_answer:
    #         group_answer = group_answer.split("**Final Answer:**")[1].strip()
    #     group_answers[i] = group_answer
        integrated_prompt = f"""
    You have a question and several LLM answers.
    Your task is to select the answer that appears most frequently (the majority answer).
    Only output the **index number** (just the number in LLM Answers, no extra words).

    Question:
    {user_prompt}

    LLM Answers:
    {" ".join(f"{i}. {d['answer']}" for i, d in enumerate(res_s))}
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
            lengths = [len(d['answer']) for d in res_s]
            group_index = min(range(len(res_s)), key=lambda i: lengths[i])

        group_answer = res_s[group_index]['answer']
        group_answers[i] = group_answer
        # print("single:", " ".join(f"{i+1}. {d['answer']}" for i, d in enumerate(res_s)))
        # print("group:", group_answer)
        # print("answer:", answer)
        # print("score:", string_pair_metrics(group_answer, answer))
        # answers = [d["answer"] for d in res_s]
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
        async_mode=True,
    )

    # 读取数据
    questions_data = load_json_from_url(data_json_path)

    if n_question_sample == "full":
        sampled_questions_data = questions_data
    else:
        sampled_questions_data = random.sample(questions_data, n_question_sample)

    sampled_user_prompts = [item["question"] for item in sampled_questions_data]
    sampled_answers = [item["answer"] for item in sampled_questions_data]

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
    for f in asyncio.as_completed(tasks):
        await f

    if n_question_sample == "full":
        scores = []
        for j in range(len(system_prompts)):
            system_prompt_answer = [majority_answers[i][j]['answer'] for i in range(len(sampled_answers))]
            performs_group = [string_pair_metrics(system_prompt_answer[i], sampled_answers[i]) for i in range(len(sampled_answers))]
            f1_mean_group = sum([p["f1"] for p in performs_group]) / len(performs_group)
            recall_mean_group = sum([p["recall"] for p in performs_group]) / len(performs_group)
            scores.append({"f1": f1_mean_group, "recall": recall_mean_group})
        # import pdb; pdb.set_trace()
        performs_group = [string_pair_metrics(group_answers[i], sampled_answers[i]) for i in range(len(sampled_answers))]
        f1_mean_group = sum([p["f1"] for p in performs_group]) / len(performs_group)
        recall_mean_group = sum([p["recall"] for p in performs_group]) / len(performs_group)
        return {"f1": f1_mean_group, "recall": recall_mean_group}, system_prompt_answer
    else:
        feedbacks = [None] * len(system_prompts)
        feedback_tasks = [
            process_feedback(i, sampled_user_prompts, sampled_answers, system_prompts, majority_answers, group_answers, lm, feedbacks)
            for i in range(len(system_prompts))
        ]
        # 并发执行，并用 tqdm 显示进度
        for f in asyncio.as_completed(feedback_tasks):
            await f
        return feedbacks, None


@dataclass
class Config(BaseConfig):
    train_json_path: str = field(default='http://cloud.staging.p1.cn/v2/ai-raw/78cffe33-ad70-428a-8205-bc2d5cd6010a.json', metadata={'desc': "feedback data path"})
    valid_json_path: str = field(default='http://cloud.staging.p1.cn/v2/ai-raw/2fee4f22-62f7-402b-9a6f-27d48cfe9eb1.json', metadata={'desc': "metric data path"})

    wiki_url: str = field(default='http://20.102.90.50:2017/wiki17_abstracts', metadata={'desc': "feedback data path"})
    top_k_search: int = field(default=10, metadata={'desc': "top k search"})
    n_question_sample: int = field(default=3, metadata={'desc': "sample nums"})

    api_key: str = field(default='', metadata={'desc': "llm api_key"})
    api_base: str = field(default='https://llm-api.p1.cn', metadata={'desc': "llm api base"})
    max_concurrency: int = field(default=10, metadata={'desc': "max concurrency"})
    model_name: str = field(default='gpt-4.1-nano', metadata={'desc': "model name"})
    temperature: float = field(default=1.0, metadata={'desc': "model temperature"})
    max_length: int = field(default=4096, metadata={'desc': "max concurrency"})

async def evaluate(program, config) -> EvalResult:
    config = Config().merge(config)
    model_name = config.model_name
    wiki_url = config.wiki_url
    api_base = config.api_base
    api_key = config.api_key
    n_question_sample = config.n_question_sample
    top_k_search = config.top_k_search
    max_length = config.max_length
    temperature = config.temperature
    train_json_path = config.train_json_path
    valid_json_path = config.valid_json_path

    if isinstance(program, str):
        program = [program]
    # assert program[0].startswith("efff"), f"{type(program)} {program}"

    #feedback
    logger.info("start get feedback")
    feedbacks, _ = await release_one_group_for_questions(
        model_name = model_name,
        wiki_url=wiki_url,
        api_base=api_base,
        api_key=api_key,
        system_prompts=program,
        top_k_search=top_k_search,
        n_question_sample=n_question_sample,
        max_length=max_length,
        temperature=temperature,
        data_json_path=train_json_path
    )

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
    eval_results.metrics = metric
    eval_results.feedback = feedbacks[0] + "==^&*(split-part)==" + "==^&*(split-val)==".join(all_final_answers)

    return eval_results
