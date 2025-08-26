SYSTEM_TEMPLATE='''
你是一名专业的审核人员，判断用户的输入内容是否违规。
'''

USER_TEMPLATE='''你的目标是让 recall 是 100% 的前提下，precision 超过 99%。

审核 input 中的消息，输出格式为：第一行为你预测该条消息是否违规，0 或者 1， 1 代表违规，0代表合规，第二行是你给出这个预测的原因。这两行内容用 <output></output> 括起来

<input>
待打标的消息内容：
${comment}

消息的上下文：
${content}

</input>
'''


import os
import re
import string
import pandas as pd
from io import StringIO
from openai import AsyncOpenAI

async def label_by_llm(system_msg, user_msg):
    client = AsyncOpenAI(
        base_url='https://llm-api.p1.cn',
        api_key=os.environ["OPENAI_API_KEY"],
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    response = await client.chat.completions.create(
        model=os.environ["MODEL_NAME"],
        messages=messages,
        temperature=1.0,
    )

    res = response.choices[0].message.content
    if res is None:
        raise ValueError(f"response is None")
    return res

def post_process(text: str):
    match = re.search(r"<output>\s*(\d+)\s*\n(.*?)\s*</output>", text, re.DOTALL)
    if not match:
        return None, None
    predicted_value = int(match.group(1))
    reason = match.group(2).strip()
    return predicted_value, reason

async def inference_function(
        batch_idx: int,
        df_input: pd.DataFrame,
):
    """
    Run a single asynchronous inference pass for the given input DataFrame.

    Parameters
    ----------
    batch_idx: int
        Just a indicator what batch this is, for later gather the results

    df_input : pd.DataFrame
        The input data containing at three columns, 'id', 'comment' and 'content'
        'id' needs to be preserved in the output.

    Returns
    -------
    batch_idx: int
        Just a indicator what batch this is, for later gather the results

    pd.DataFrame or None
        The processed output DataFrame with the same 'id' values as the input, it contains two columns: 'id' and 'predicted' at least
        or None if the inference fails after multiple retries.

    raw_res: string
        raw string response from llm

    """
    assert len(df_input) == 1
    id = df_input['id'].iloc[0]
    comment = df_input['comment'].iloc[0]
    content = "\n".join(reversed(df_input['content'].iloc[0].splitlines()))

    user_tpl = string.Template(USER_TEMPLATE)
    user_args = {
                'comment': comment,
                'content': content,
    }
    system_message = SYSTEM_TEMPLATE
    user_message = user_tpl.safe_substitute(user_args)

    raw_res = await label_by_llm(
        system_message,
        user_message,
    )

    times_try = 0
    while True:
        try:
            predicted, reason = post_process(raw_res)
            df_output = pd.DataFrame([[id, predicted, reason]],
                    columns=["id", "predicted", "reason"])
            
            merged = pd.merge(df_input, df_output, on='id', how='inner')

            return batch_idx, merged, raw_res

        except Exception as e:
            times_try += 1
            print(f"Batch {batch_idx + 1} failed, attempt {times_try}/5: {e}")
            if times_try >= 5:
                return batch_idx, None, raw_res