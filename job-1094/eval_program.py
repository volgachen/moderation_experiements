
import re
import os
import asyncio
from asyncio import CancelledError
import string
from io import StringIO
import random
import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict

import pandas as pd
import numpy as np
from openai import AsyncOpenAI, BadRequestError
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_not_exception_type, retry_if_exception_type

from alpha_evolve_evaluator.evaluator import EvalResult, BaseConfig
from alpha_evolve_evaluator.dataset import Dataset, keep_columns

"""
Minimal client for calling embeddings API using Python standard library only.

Function
--------
get_embeddings(inputs, dim=2048, timeout=30.0)
    Calls POST http://10.10.1.124:8001/api/embeddings with the given inputs.
    - inputs: str or list[str]
    - dim: keep only the first N dimensions from {2048, 1024, 512, 256}
    - returns: list[list[float]] with the same length as inputs (1 if inputs is str)

Example
-------
    vecs = get_embeddings(["text 1", "text 2"], dim=512)
    print(len(vecs), len(vecs[0]))
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Literal, Sequence, Union


def get_embeddings(
    inputs: Union[str, Sequence[str]],
    dim: Literal[2048, 1024, 512, 256] = 2048,
    timeout: float = 30.0,
) -> List[List[float]]:
    """
    Call Jarvis embeddings API and return vectors with the same length as inputs.

    Args:
        inputs: a single text or a list of texts
        dim: keep only the first N dimensions; must be one of {2048, 1024, 512, 256}
        timeout: request timeout in seconds

    Returns: embedding vectors with the same length as inputs
    """

    def _normalize_inputs(inputs: Union[str, Sequence[str]]) -> List[str]:
        """
        Normalize inputs to a list of strings.
        """
        if isinstance(inputs, str):
            return [inputs]
        # Ensure all items are strings
        return [s if isinstance(s, str) else str(s) for s in inputs]

    base_url = "http://10.10.1.124:8001"

    texts = _normalize_inputs(inputs)

    if dim not in (2048, 1024, 512, 256):
        raise ValueError("dim must be one of {2048, 1024, 512, 256}")

    url = f"{base_url.rstrip('/')}/api/embeddings"
    payload: Dict[str, Any] = {"input": texts}
    if dim is not None:
        payload["dim"] = dim

    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_text = resp.read().decode("utf-8")
            status = getattr(resp, "status", 200)
            if status < 200 or status >= 300:
                raise RuntimeError(f"HTTP {status}: {resp_text}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
        raise RuntimeError(f"HTTPError {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URLError: {e}") from e

    try:
        data = json.loads(resp_text)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse response: {e}; raw: {resp_text[:2000]}"
        ) from e

    items = data.get("data", [])
    # Sort by index defensively to align order
    try:
        items = sorted(items, key=lambda x: int(x.get("index", 0)))
    except Exception:
        pass

    result: List[List[float]] = []
    for item in items:
        vec = item.get("embedding")
        if not isinstance(vec, list):
            vec = []
        # Ensure a list of floats
        try:
            result.append([float(x) for x in vec])
        except Exception:
            # If conversion fails, keep the original list
            result.append(vec)

    return result


_logger = logging.getLogger(__name__)

def setup_stdout_logging(log_level: str='INFO', logger_names: list[str]=['evolver']) :
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.addHandler(handler)


setup_stdout_logging(log_level="INFO", logger_names=[__name__])

@dataclass
class Config(BaseConfig):
    train_data_path: str = field(default='http://cloud.staging.p1.cn/v2/ai-raw/e2901005-5939-4b9f-ac56-5790b6a71560.parquet', metadata={'desc': "train data path"})
    validate_data_path: str = field(default='http://cloud.staging.p1.cn/v2/ai-raw/8a39f1f6-6a5e-468c-94a8-b6a42ced63d1.parquet', metadata={'desc': "validate data path"})
    test_data_path: str = field(default='http://cloud.staging.p1.cn/v2/ai-raw/45634760-c8a8-417a-9557-c492687d2ed8.parquet', metadata={'desc': "test data path"})

    sample_nums: int = field(default=500, metadata={'desc': "sample nums"})
    # batch_size: int = field(default=50, metadata={'desc': "batch size"})
    metric_type: str = field(default='f1', metadata={'desc': "f1, precision+recall"})
    max_concurrency: int = field(default=10, metadata={'desc': "max concurrency"})

    model_name: str = field(default='gpt-4.1-mini', metadata={'desc': "model name"})
    model_temperature: float = field(default=1.0, metadata={'desc': "model temperature"})
    api_key: str = field(default='sk-VseOo3-mYm7kBTosV2L1Gg', metadata={'desc': "llm api_key"})

    test: bool = field(default=False, metadata={'desc': "run on test dataset"})
    seed: int = field(default=-1, metadata={'desc': "random seed"})
    different_seed: int = field(default=False, metadata={'desc': "each run use different seed"})
    log_dir: str = field(default='logs/user-messages-new/', metadata={'desc': "log directory"})


@retry(
        wait=wait_fixed(1),
        stop=stop_after_attempt(3),
        reraise=True,
        # before_sleep=before_sleep_log(_logger, logging.DEBUG),
        retry=retry_if_not_exception_type((BadRequestError, CancelledError)),
        )
async def label_by_llm(client, system_msg, user_msg, conf: Config):
    _logger.debug(f"model: {conf.model_name}, temperature: {conf.model_temperature}", )
    _logger.debug("system message: \n%s", system_msg)
    _logger.debug("user message: \n%s", user_msg)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # model = "openrouter/anthropic/claude-sonnet-4"
    # model = "vertex_ai/gemini-2.5-flash"
    model = conf.model_name
    temperature = conf.model_temperature
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        # max_tokens=8192,
        # top_p=1.0,
    )

    res = response.choices[0].message.content
    if res is None:
        raise ValueError(f"response is None")
    return res

def exp_weighted(x, k=3):
    return (np.exp(k * x) - 1) / (np.exp(k) - 1)

class ExtractDfError(Exception):
    """Raised when cannot extract csv from llm output occurs."""
    pass


def format_df(df: pd.DataFrame, all_strs, threshold: float=1.0):
    out_FP = []
    out_FN = []
    out_TN = []
    out_TP = []
    format = '''# 示例样本{id}
## 待审核消息: {message}

## 消息所在对话: \n{context}\n

## 模型原始输出文本: 
{result}

## 模型预测结果: {predicted}

## 正确答案: {ground_truth}

'''

    for idx, row in df.iterrows():
        predicted = row['predicted']
        if predicted is None:
            continue
        predicted = 1.0 if predicted >= threshold else 0.0
        out = format.format(
            id=row['id'],
            message=row['comment'],
            # context=row['content'],
            context="\n".join(reversed(row.get('content', '').splitlines())),
            ground_truth="违规 (1)" if row['label'] else "正常 (0)",
            predicted="违规 (1)" if row['predicted'] else "正常 (0)",
            result=all_strs[idx],
        )

        if predicted == 0 and row['label'] == 1:
            out_FP.append(out)
        elif predicted == 1 and row['label'] == 0:
            out_FN.append(out)
        elif predicted == 1 and row['label'] == 1:
            out_TP.append(out)
        elif predicted == 0 and row['label'] == 0:
            out_TN.append(out)

#     return f'''
# 违规但被错误预测为不违规的(False Positive)(共计 {len(out_FP)} 条):
# {'\n'.join(out_FP)}
# 不违规但被错误预测为违规的(False Negative)(共计 {len(out_FN)} 条):
# {'\n'.join(out_FN)}
# 不违规且被正确预测为不违规的(True Negative)(共计 {len(out_TN)} 条):
# {'\n'.join(out_TN)}
# 违规且被正确预测为违规的(True Positive)(共计 {len(out_TP)} 条):
# {'\n'.join(out_TP)}
#     '''
    sampled_FP = random.sample(out_FP, min(len(out_FP), 5))
    if len(out_FP) > 10:
        sampled_FP.append('其余的省略。。。')
    sampled_FN = random.sample(out_FN, min(len(out_FN), 5))
    if len(out_FN) > 10:
        sampled_FN.append('其余的省略。。。')
    sampled_TP = random.sample(out_TP, min(len(out_TP), 3))
    if len(out_TP) > 10:
        sampled_TP.append('其余的省略。。。')
    sampled_TN = random.sample(out_TN, min(len(out_TN), 3))
    if len(out_TN) > 10:
        sampled_TN.append('其余的省略。。。')

    return f'''
具体打标详情如下，每个分类最多显示10条。

违规但被错误预测为不违规的(False Positive)(共计 {len(out_FP)} 条):
{'\n'.join(sampled_FP)}
不违规但被错误预测为违规的(False Negative)(共计 {len(out_FN)} 条):
{'\n'.join(sampled_FN)}
不违规且被正确预测为不违规的(True Negative)(共计 {len(out_TN)} 条): 
{'\n'.join(sampled_TN)}
违规且被正确预测为违规的(True Positive)(共计 {len(out_TP)} 条):
{'\n'.join(sampled_TP)}
    '''


def format_df1(df: pd.DataFrame):
    out = ''
    format = ''' # {id}
message: {message}
message context: \n{context}\n
ground_truth: {ground_truth}

'''

    for _, row in df.iterrows():
        out += format.format(
            id=row['id'],
            message=row['comment'],
            # context=row['content'],
            context="\n".join(reversed(row.get('content', '').splitlines())),
            ground_truth=row['label'],
        )

    return out

def calculate_metrics(dataset, res_df: pd.DataFrame, metric_type: str, sample_nums: int) -> Dict[str, float]:
    precision, recall = dataset.precision_n_recall(
        res_df,
        predicted_label_name="predicted",
    )
    precision = round(float(precision), 4)
    recall = round(float(recall), 4)

    if metric_type == 'f1':
        f1 = round(float(2*precision*recall/(precision+recall)), 4) if precision+recall > 0 else 0.0
        # f1_exp = round(float(exp_weighted(f1, k=3)), 4)
        metrics = {
            'f1_score': f1,
            # 'score': f1_exp,
            # 'f1': f1,
        }
    else:
        metrics = {
            'precision': precision,
            'recall': recall,
            'score': round(float(3*precision + recall), 4),
        }

    labeled_ratio = res_df.shape[0] / sample_nums
    _logger.info(f"total: {sample_nums}, labeled: {res_df.shape[0]}, labeled_ratio: {labeled_ratio}")
    
    for k, v in metrics.items():
        metrics[k] = v * labeled_ratio

    metrics['labeled_ratio'] = round(labeled_ratio, 4)
    return metrics

async def process(conf: Config, validate: bool):
    if conf.test:
        path = conf.test_data_path
    else:
        path = conf.validate_data_path if validate else conf.train_data_path

    dataset = Dataset(
        path=path,
        type="parquet",
        truth_column_name="label",
    )

    if conf.seed != -1:
        _logger.info(f"set random seed to {conf.seed}")
        random.seed(conf.seed)
        np.random.seed(conf.seed)

    if conf.test or validate:
        input_df = dataset._df
    else:
        sample_nums = conf.sample_nums
        input_df = dataset.sample(n=sample_nums, no_repeat=True)

    sem = asyncio.Semaphore(conf.max_concurrency)

    # batch_size = max(1, conf.batch_size)
    batch_size = 1
    batches: List[pd.DataFrame] = [input_df.iloc[i : i + batch_size] for i in range(0, len(input_df), batch_size)]

    os.environ["OPENAI_API_KEY"] = conf.api_key
    os.environ["MODEL_NAME"] = conf.model_name

    tasks = [
        asyncio.create_task(inference_function(i, b))
        for i, b in enumerate(batches)
    ]
    all_results: List[pd.DataFrame] = [None] * len(batches)
    all_raw_texts: List = [None] * len(batches)

    early_stopped = False
    finished = 0

    cancel_unfinished = lambda: [t.cancel() for t in tasks if not t.done()]    

    for fut in asyncio.as_completed(tasks):
        idx, df_chunk, raw_text = await fut
        finished += 1

        all_results[idx] = df_chunk
        all_raw_texts[idx] = raw_text

        # check whether we need early stopping
        # if finished % conf.max_concurrency == 0:
        #     early_stopped = True

        # if False:
        #     cancel_unfinished()
        #     break
    
    res_df = pd.concat(all_results, ignore_index=True)
    # save the res_df to log dir
    res_df.to_csv(os.path.join(conf.log_dir, 'result.csv'), index=False)
    if validate:
        metrics = calculate_metrics(dataset, res_df, conf.metric_type, input_df.shape[0])
        return metrics
    else:
        # 反馈样本数量信息
        metrics = calculate_metrics(dataset, res_df, conf.metric_type, input_df.shape[0])
        positive_count = res_df[res_df['label'] == 1].shape[0]
        all_count = res_df.shape[0]
        feedback = f"总样本数: {all_count}, 其中违规的样本数量: {positive_count}, 占比: {round(positive_count/all_count, 4)}\n"

        feedback += format_df(res_df, all_raw_texts)
        return feedback

async def evaluate(program: str, config: Config) -> EvalResult:
    """
    Evaluate the evolved program.
    
    Args:
        program: The evolved program code
        config: The config object
        
    Returns:
        EvalResult with metrics and feedback
    """
    conf = Config().merge(config)
    _logger.info(f"evaluate program with config: {conf}")
    if not os.path.exists(conf.log_dir):
        os.makedirs(conf.log_dir)
    else:
        _logger.info(f"{conf.log_dir} already exists ...")
    # system_msg, user_msg, new_code = extract_msg(program)
    exec(program, globals())
    ans = EvalResult()
    ans.metrics['f1_score'] = 0.0  # make sure return all metrics defined in config.yaml

    if conf.test:
        ans.metrics = await process(conf, validate=True)
    else:
        ans.metrics, ans.feedback = await asyncio.gather(
            process(conf, validate=True),
            process(conf, validate=False)
        )

    return ans