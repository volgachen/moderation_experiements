import requests
import dspy
import json
import re
import random
import asyncio
import time
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

def safe_retrieve(retrieve_fn, query, retries=200, delay=1):
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
        self.create_query_hop2 = dspy.ChainOfThought("question,summary_1->query")
        self.final_answer = dspy.ChainOfThought("question,summary_1,summary_2->answer")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("question,passages->summary")
        self.summarize2 = dspy.ChainOfThought("question,context,passages->summary")

    def forward(self, question):
        # HOP 1
        hop1_docs = safe_retrieve(self.retrieve_k, question)
        hop1_docs = ' '.join([str(i+1)+". " + hop1_docs[i] for i in range(len(hop1_docs))])
        try:
            summary_1 = self.summarize1(
                question=question, passages=hop1_docs
            ).summary  # Summarize top k docs
        except:
            summary_1 = "Warning: Repetition phenomenon occurred, output is incomplete."

        # HOP 2
        try:
            hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        except:
            hop2_query = "Warning: Repetition phenomenon occurred, output is incomplete."

        hop2_docs = safe_retrieve(self.retrieve_k, hop2_query)
        hop2_docs = ' '.join([str(i+1)+". " + hop2_docs[i] for i in range(len(hop2_docs))])
        try:
            summary_2 = self.summarize2(
                question=question, context=summary_1, passages=hop2_docs
            ).summary
        except:
            summary_2 = "Warning: Repetition phenomenon occurred, output is incomplete."

        try:
            answer = self.final_answer(
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
    res = await asyncio.to_thread(system, user_prompt)
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
    successed = 0
    for f in asyncio.as_completed(tasks):
        await f
        successed += 1
        logger.info(f"Successed: {successed}")

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



def build_group_usr_msg(question, val_per_question):
    integrated_prompt = f"""
        You have a question and several LLM answers.
        Your task is to select the answer that appears most frequently (the majority answer).
        Only output the **index number** (just the number in LLM Answers, no extra words).

        Question:
        {question}

        LLM Answers:
        {" ".join(f"{i}. {d}" for i, d in enumerate(val_per_question))}
    """
    return integrated_prompt

def select_index_answer(llm_res_index, val_per_question):
    if llm_res_index == None:
        llm_res_index = "Warning: Repetition phenomenon occurred, output is incomplete."

    llm_res_index = llm_res_index[0].strip()   # 去掉前后空格换行

    # 提取第一个纯数字
    match = re.search(r"\d+", llm_res_index)
    if match and int(match.group()) in range(len(val_per_question)):
        llm_res_index = int(match.group())
    else:
        # fallback: 选择最短的答案
        lengths = [len(d) for d in val_per_question]
        llm_res_index = min(range(len(val_per_question)), key=lambda i: lengths[i])
    group_answer = val_per_question[llm_res_index]
    return group_answer


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

async def evaluate_single(program, config) -> EvalResult:
    parent_program = config["insights"][0]
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
    eval_results.metrics = {"f1": parent_program["metrics"]["f1"], "recall":  parent_program["metrics"]["recall"]}
    eval_results.feedback = "None" + "==^&*(split-part)==" + "==^&*(split-val)==".join(all_final_answers)

    return eval_results


async def evaluate_group(program, config) -> EvalResult:
    val_results = config.pop("val_results")
    programs = config.pop("programs")

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

    questions_data = load_json_from_url(valid_json_path)
    sampled_user_prompts = [item["question"] for item in questions_data]
    sampled_answers = [item["answer"] for item in questions_data]

    client = AsyncOpenAI(
        base_url='https://llm-api.p1.cn',
        api_key=config.api_key,
    )

    @retry(
        wait=wait_fixed(1),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def process_one(i):
        val_per_question = [val_results[j][i] for j in range(len(val_results))]
        group_usr_msg = build_group_usr_msg(sampled_user_prompts[i], val_per_question)
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a answer selector"},
                {"role": "user", "content": group_usr_msg},
            ],
            temperature=temperature,
        )
        llm_res_index = response.choices[0].message.content
        group_answer = select_index_answer(llm_res_index, val_per_question)
        metric = string_pair_metrics(group_answer, sampled_answers[i])
        return metric

    tasks = [
        asyncio.create_task(process_one(i))
        for i in range(len(val_results[0]))
    ]

    metrics = []
    for fut in asyncio.as_completed(tasks):
        metric = await fut
        metrics.append(metric)
    f1_mean_group = sum([p["f1"] for p in metrics]) / len(metrics)
    recall_mean_group = sum([p["recall"] for p in metrics]) / len(metrics)
    final_metrics = {"f1": f1_mean_group, "recall": recall_mean_group}

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


if __name__ == "__main__":
    program = '''<system_prompt_1>
You are the first-hop **summarization module** in a multi-hop QA system. Your task is to generate a **comprehensive, structured summary** that:  1. **Extracts direct answers** from the top retrieved passages to address the question. 2. **Identifies and highlights missing or implied clues** that may require further retrieval (e.g., entities, connections, or contextual details).3. **Synthesizes information** by combining explicit facts from the passages with domain-specific knowledge or logical inferences to guide subsequent steps.  ### **Summary Structure** - **Entity/Person Mention**: Clearly state the subject (e.g., "Billy Truax", "Eintracht Braunschweig") and include **full names, titles, or official designations** (e.g., "Thomas Lance Rentzel", "Braunschweiger Turn- und Sportverein Eintracht von 1895 e.V."). - **Direct Answer**: Include **explicit answers** from the passages (e.g., birth dates, team affiliations, or direct statements). - **Clues for Next Steps**: Signal **missing information** (e.g., "Lance Rentzel's birth year is explicitly stated, but his exact birthplace is not; need to search for 'Lance Rentzel birthplace'"). - **Domain-Specific Context**: Add **relevant background** (e.g., "Eintracht Braunschweig is a German football club based in Braunschweig, Lower Saxony" or "NFL players' birth dates are critical for age comparisons").  ### **Guidelines** - **Do not omit** any entity or detail from the retrieved passages that could be relevant for follow-up queries (e.g., team names, locations, or historical context). - **Prioritize clarity** by **separating direct answers from inferred clues** (e.g., using bullet points, subheadings, or bolded labels). - **Avoid assumptions** not supported by the passages; if information is absent, **explicitly state that it is missing** and suggest **precise search terms** (e.g., "Verify Wichita Dwight D. Eisenhower National Airport's tower status via FAA records"). - **Include quantifiable data** (e.g., "few thousand Stabyhouns exist globally", "born July 15, 1943") to enable precise comparisons. - **Highlight connections** between entities (e.g., "Billy Truax and Lance Rentzel were traded in 1970") to aid in cross-referencing.  ### **Key Niche/Domain-Specific Insights** - **NFL Player Comparison**: Birth dates are critical for age determination, and team affiliations (e.g., "traded in 1970") may imply historical context. - **Airport Classification**: "Non-towered" status is explicitly stated in some passages (e.g., "non-towered public airport"), while others require inference (e.g., "major commercial airports typically have towers"). - **Football Club Context**: Clubs like Eintracht Braunschweig require background on their location, league, and history (e.g., "based in Braunschweig, Lower Saxony"). - **Quantifiable Data**: Use exact dates, numbers, or rankings (e.g., "few thousand Stabyhouns exist globally") to enable precise comparisons.  ### **Critical Additional Instructions** - **Ensure All Retrieved Documents Are Represented**: Explicitly include all entities, titles, and details from the retrieved passages (e.g., full names, film titles, and specific roles). - **Signal Missing Links**: If a connection between entities is implied but not explicitly stated (e.g., "Nancy Steiner worked on *The Lovely Bones*"), flag this as a potential gap and suggest search terms to resolve it. - **Prioritize Bridging Concepts**: Highlight relationships between entities (e.g., "Gary Pinkel coached Toledo in 1993 and holds the most wins in school history") to enable focused follow-up queries. - **Avoid Overgeneralization**: Only include domain-specific context that is either explicitly stated in the passages or directly inferable (e.g., "major commercial airports typically have towers" is acceptable, but "airports with fewer than 10,000 passengers are non-towered" is not unless stated).  ### **Example Format** For the question *"Which NFL player is younger, Billy Truax or Lance Rentzel?"*: - **Entity/Person Mention**: Billy Truax (William Frederick Truax), Lance Rentzel (Thomas Lance Rentzel)- **Direct Answer**: - **Billy Truax**: Born July 15, 1943. - **Lance Rentzel**: Born October 14, 1943. - **Clues for Next Steps**: None required; birth dates are explicitly provided. - **Domain-Specific Context**: Birth dates are sufficient to determine age difference within the same year.  For the question *"Which is a non-towered airport, Wichita Dwight D. Eisenhower National Airport or Montrose Regional Airport?"*: - **Entity/Person Mention**: Wichita Dwight D. Eisenhower National Airport, Montrose Regional Airport - **Direct Answer**: - **Montrose Regional Airport**: "non-towered public airport" (passage 3). - **Wichita Dwight D. Eisenhower National Airport**: No explicit mention of tower status; inferred as **towered** (typical for major commercial airports). - **Clues for Next Steps**: Verify Wichita's tower status via FAA records or additional sources (e.g., "Wichita Dwight D. Eisenhower National Airport tower status"). - **Domain-Specific Context**: Non-towered airports lack a control tower, relying on pilot communication (passage 4). Major commercial airports like Wichita usually have towers.  **Tip:** When summarizing, dont just compress; synthesizeinclude both direct answers and clues required for the systems next steps. Always explicitly state if a retrieved documents content is missing critical information, and provide actionable search terms to address gaps.
</system_prompt_1>

<system_prompt_2>
Given the fields 'question' and 'summary_1', produce the field 'query' that optimizes the retrieval of additional documents for a multi-hop system.  **Task Details:** 1. **Objective:** Your query must target documents not retrieved in the first hop, using clues from the summary and the original question. 2. **Key Strategy:** - Identify gaps in the first hop's retrieved documents (e.g., missing entities, relationships, or specific details). - Use explicit information from the summary (e.g., names, locations, quantities) to rephrase the question into a query that surfaces new relevant documents. - Avoid restating the answer directly; instead, structure the query to explore connections or unresolved details. 3. **Domain-Specific Guidance:** - If the summary explicitly answers the question, the query should still focus on retrieving documents that provide deeper context or verify the answer (e.g., "What is the headquarters location of [Company]?" instead of "The answer is [Location]"). - Leverage entities mentioned in the summary (e.g., "Carhartt," "Aubrey O'Day") to anchor the query. - If no documents are missing, rephrase the query to explicitly request the answer (e.g., "Which has more acts, Elektra or From the House of the Dead?"). 4. **Avoid:** - Generating queries that duplicate the original question. - Assuming the summary contains all necessary information for the second hop.
</system_prompt_2>

<system_prompt_3>
Given the fields 'question', 'context', and 'passages', produce the field 'summary'.  Your task is to synthesize information from the question, context, and newly retrieved passages to generate a **comprehensive, precise, and well-structured summary** that enables the answer generation module to confidently arrive at the correct answer.  ### Key Requirements: 1. **Explicit Answers First**: Prioritize explicitly stated facts from the context and passages (e.g., direct mentions of entities, roles, or relationships). 2. **Infer or Generalize When Necessary**: If critical details are missing from the passages, infer connections or generalize based on contextual clues and domain-specific knowledge (e.g., linking ownership structures, roles, or historical context). 3. **Bridge Gaps**: Ensure the summary includes all **key supporting information** required to answer the question, even if it is not explicitly stated in the input. For example: - If the answer is "Newcastle United," include details about Sports Direct's ownership and the connection to the billionaire. - If the answer is a person's role (e.g., "troubleshooter"), explicitly state their relationship to the question's subject and any relevant background. 4. **Structure and Precision**: - Clearly connect entities, roles, and relationships (e.g., "Stan Kroenke owns Sports Direct and Arsenal F.C."). - Avoid ambiguity by including all necessary contextual links (e.g., "Mike Ashley founded Sports Direct and owns Newcastle United"). - Use precise terminology and ensure alignment with domain-specific knowledge (e.g., "investigative journalist" instead of "writer"). 5. **Domain-Specific Knowledge**: Leverage implicit domain knowledge when passages lack critical details (e.g., knowing that "Project RAND" is linked to Henry H. Arnold and the RAND Corporation).  ### Example Integration: If the question is about a person's profession in a novel, ensure the summary includes: - The character's name. - Their profession (explicitly stated in the text). - Contextual links to the book series or plot (e.g., "in *The Girl in the Spider's Web*"). - Any relevant background about the profession or characters role in the story.  Always aim to match the **coverage and relevance** of an "ideal summary" as described in the feedback, ensuring the answer module has all necessary information to generate the correct final answer.
</system_prompt_3>

<system_prompt_4>
Given the fields 'question', 'summary_1', and 'summary_2', produce the field 'answer' by: 1. **Extracting precise terminology**: Identify the exact noun or specific term required in the answer (e.g., "Medicare" rather than "Medicare cuts"). Avoid vague or generalized terms unless explicitly stated in the summaries. 2. **Resolving ambiguity**: If the question references a title, historical role, or specific designation (e.g., "second Duke of Florence"), prioritize contextual or historical clues from the summaries to infer the correct answer, even if the exact term is not explicitly stated. Use domain-specific knowledge (e.g., Medici family lineage) to fill gaps when summaries are indirect or vague. 3. **Cross-referencing summaries**: Ensure consistency between summaries. If summaries conflict, prioritize the one with explicit factual claims (e.g., numerical data, direct statements). If no explicit claim exists, synthesize information while ensuring alignment with historical, political, or cultural context. 4. **Avoiding overgeneralization and extra information**: Focus strictly on the most specific and directly stated information in the summaries. Do not add context, explanations, or external knowledge beyond what is explicitly provided. For example, if the question asks for a year, provide only the year; do not include band member details or historical background. 5. **Prioritizing factual alignment**: If a summary explicitly states the answer, use that. If summaries are indirect or vague, synthesize information while ensuring alignment with factual knowledge (e.g., linking "Path to Prosperity" to Rep. Paul Ryans Medicare proposal).  **Key adjustments based on feedback**: - **Conciseness**: Answers must be strictly factual and concise, avoiding additional context or explanations. For example, if the question is "Is X shorter than Y?" the answer should be a simple "No" or "Yes" based on numerical comparisons, not a full explanation. - **Numerical precision**: When comparing measurements (e.g., heights, dates), ensure exact values are used and explicitly stated in the summaries. If summaries provide conflicting numbers, resolve via direct factual claims. - **Domain-specific knowledge**: Use known facts (e.g., architectural records, historical timelines) to validate ambiguous answers, but only when summaries lack explicit information.
</system_prompt_4>'''
    res = asyncio.run(evaluate(program, {
        "valid_json_path": "http://cloud.staging.p1.cn/v2/ai-raw/6cbd4a88-404b-4fa3-9ba5-65c887a6b336.json",
        "enable_feedback": True,
        "programs": [program, program, program],
        "val_results": [["", "", "", "", ""], ["", "", "", "", ""], ["", "", "", "", ""]],
    }))
    import pdb; pdb.set_trace()