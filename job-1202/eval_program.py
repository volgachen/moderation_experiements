
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


RAW_PROGRAM = ""


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


sem = asyncio.Semaphore(50)  # at most 50 tasks at a time

@retry(
        wait=wait_fixed(1),
        stop=stop_after_attempt(3),
        reraise=True,
        # before_sleep=before_sleep_log(_logger, logging.DEBUG),
        retry=retry_if_not_exception_type((BadRequestError, CancelledError)),
        )
async def get_reflection(df_input, all_quick_overview):
    # withe the provided 
    system_message = """你是一名程序分析师，正在分析一个用户聊天行为审核程序的好坏。程序的目标是审查用户在牵手异性交友平台上聊天时，某一句指定聊天内容是否存在违规行为。违规样本应该被判定为正样本 (1, positive) ，正常样本应该被判定为负样本 (0, negative) 。程序开发的最终目标应当与人类审核员的审核结果高度对齐，从而能代替人类审核员直接开展审核工作。

在牵手（该聊天平台的名称）的标注人员的培训手册上，标注了以下这些违规类型。这一培训资料可以用作参考，帮助你更好地理解人类审核员的评判标准。但我们不保证培训手册完全准确，如有冲突，以人类审核员提供的审核结果为准。

违规类型：攻击党政反动分裂
判断标准：涉政行为底线：
1.宣扬、煽动有分裂意图及破坏国家统一的内容。例如：宣扬民族独立，煽动民族仇恨、民族歧视，破坏民族团结
（1）台独/港独/疆独/藏独/蒙独/沪独等分裂言论和相关的事件报道和内容
（2）涉及民族独立（台独/港独/疆独/藏独/蒙独/沪独）相关信息
2.恶搞、丑化、侮辱、讽刺、否定领导人/英雄烈士/功勋人物事迹和精神、言论、行为等
3.对底线敏感事件持支持、宣传、鼓励态度，或存在反政府、隐喻讽刺社会政策等相关内容
4.宣扬、号召加入极端组织，例如ISIS、基地组织、纳粹及其他明令禁止的组织
5.传播境外媒体/社交软件发布政治有害/可疑内容（与政府、社会、宗教、民族等相关的非正面内容）
6.对底线敏感事件持支持、宣传、鼓励态度，或存在反政府、隐喻讽刺社会政策等相关内容。
（1）涉港台抗议、游行、暴力等热点事件描述、转述、讨论内容（如香港占中、返送中、太阳花运动及其他）
（2）涉反动暴乱性质相关的人物、网站、图书（64事件、法轮功、75事件、314事件、709律师及其他）
详细管控名单@反动媒体 
（3）散布谣言，煽动维权上访/群体性事件等聚众扰乱社会秩序，破坏社会稳定的行为（佳士工人维权、老兵维权及其他）

违规类型：攻击党政反动分裂
判断标准：涉政人物底线：
1.恶搞、丑化、侮辱或讽刺在职/去职副部级以上领导 
（1）讨论领导人及其家属相关的野史、内幕的内容 领导人家属 
（2）以历史人物、历任皇帝隐喻、讽刺、攻击 在职/去职领导人
（3）讽刺、调侃、恶搞部级领导人及国家执法机关的漫画内容
2.歪曲、丑化、亵渎、否定英雄烈士/功勋人物事迹和精神，以侮辱、诽谤或者其他方式侵害英雄烈士/功勋人物
功勋人物英雄烈士名录 
3.底线/异见人士 —— 涉政人物底线（内部资料严禁外传，请遵守保密协议） 
（1）台独/港独/疆独/藏独/蒙独/沪独相关人物
（2）底线异见人士、敏感人物
4.注册账号头像为在职/去职副国级以上领导人，一级主管部门领导人、有重大社会和舆论影响或副部级以上的落马官员"

违规类型：违法
判断标准：1.宣扬、引诱、吸食毒品，或展示实物毒品、描述吸毒细节、毒品效果
2.代孕等违法行为
3.传销性质微商（同公开信息名单）

违规类型：索要红包礼物
判断标准：站内站外向一位用户索要财物-借钱、索要红包，或提及交友涉及金钱交易和交换（特殊节日红包、批判驳斥及仅提及红包除外），例如：要外卖、加微信需要发红包（奖励）、要打车费、要化妆品、索要物品等情形同红包规则处理

违规类型：索要红包礼物
判断标准：站内站外向多位（≥2）用户索要财物-借钱、索要红包，或提及交友涉及金钱交易和交换（特殊节日红包、批判驳斥及仅提及红包除外），例如：要外卖、加微信需要发红包（奖励）、要打车费、要化妆品、索要物品等情形同红包规则处理
备注：
①坐实：举报项有完整证据链（两个及以上不同用户）
情况1：实锤举报 + 2条以上不同人的历史举报（必须是要红包、诈骗类型的举报标签）；情况2: 实锤举报 + 百条有1条以上站内要红包（不能与当前举报人为同一ID）
②以上两种情况都需要满足不同举报人
③仅请客吃饭类无法判断线上/线下的不做处罚，例如：请我吃饭（男用户不处罚，女用户正常处罚）
④索要金额超3千元直接诈骗封禁
⑤男方主动女方调侃性或玩笑语境下索要召回/人工不管控

违规类型：金融诈骗
判断标准：1.网投、网彩、时时彩、牛牛、龙虎、百家乐、北京28、加拿大28、赌球、竞猜等赌博性质的游戏、彩票、看走势、行情等（非实体店类彩票均ban）、影视宣发、影视传媒、非合法银行借贷拉客
2.充电宝诈骗（向他人请求帮扫充电宝，到时不还）

违规类型：未成年
判断标准：明确表明自己是未成年人（18 岁以下），例如表明自己是高中生、初中生或16岁、17岁

违规类型：已婚
判断标准：被举报人聊天中提及自己已婚
举报人提供明确证明对方已婚且在婚姻状态中证据，例如：结婚证、结婚照等

违规类型：泄密
判断标准：聊天过程种表明自己为牵手恋爱/红娘员工，有以下任一情况：
1.对公司业务进行详细描述
2.产生威胁话语，或套取个人隐私

违规类型：非本人账号
判断标准：私聊中提及当前账号是非本人的，例如：这个是我家其他人的注册的；这个账号是我妹妹的或者其他人的
表明给其他亲戚、朋友、同事找对象的，例如：我来帮我哥哥（姐姐）找对象

违规类型：隐晦色情
判断标准：1.随意评价对方的身材长相，例如：你好丰满、前凸后翘
2.玩梗类的擦边行为，例如：萝莉有三宝，清纯，可愛，易推倒；空虚寂寞冷；处暧昧关系吧；贪财好色；
备注：玩梗类的百条向≥3人发送，按禁言处罚
3.星座相关的非指向性擦边形容词，例如：XX星座欲望大；XX星座喜欢找刺激
4.非恶意色情图片，例如：聊天中出现微信背景黄图（男女露点、勃起内裤照、性行为、女性乳晕），绘画作品中出现裸露图片等
5.处男处女，讨论的目标不是对方或自己
6.下腹部到敏感部位的图片、视频（仅限人工举报，召回不处罚）
7.胸/屁股/舌头/脚此类敏感部位的夸赞（例：你的屁股练得好漂亮、你的脚很好看）
提醒级别色情词汇召回给自见，人工举报提醒处罚

违规类型：隐晦色情
判断标准：1.前后语境容易引发色情联想的行为或有歧义的言语，例如：晚上一个人寂寞吗；带回家就不让你走了；班有我好上吗；我想和你去私人影院；喜欢做核酸吗；能和你玩玩吗；想男人了吗；我想跟你生孩子；你内衣是什么颜色；该大的地方大；我很色/你好色吗；我喜欢胸/屁股大的；你是来约炮的吗；活好吗；你约过吗；撕丝袜；聊聊男女之事；
备注：色情联想类的语句需要结合前后语境来判断，不做过度联想
2.讨论、形容敏感部位（腿、脚等），例如：看看腿（脚）；你穿丝袜吗；看看hs照可以吗；我是恋足癖（丝袜恋足类：人工举报处罚，关键词召回不处罚）
3.婴幼儿生殖器、隐晦表情包：图没问题但是配的文字有隐晦色情内容
4.约吗、做吗等无明显指向性的隐晦性邀约行为（百条里向≥3人发送此类会话，可直接升级禁言处理）
5.讨论性观点例如：你怎么看待性；你觉得性和爱哪个更重要；我觉得性和谐很重要；你怎么看待两性关系；离婚这么久会性冷淡吗；
6.sm类自述身份或者询问对方，例如：你是sub吗；我是字母圈的  备注sm类：如果用户是因为举报人先发送容易引发联想的语句导致被召回，不做处罚
备注：1.若百条内能确认用户玩软件的目的是yp、找py等，可升级ban处罚 2.若警告级别色情词刷屏≥3人可升级禁言处罚；

违规类型：轻度色情
判断标准：1.发未露生殖器的低俗类图片/视频
2.发色情暗示的音频信息，如：娇喘、叫床、色情片音频片段等
3.涉及隐晦金钱的邀约，如：500块钱出来约吗；
4.询问对方性生活历史或大尺度描述自身性生活历史的行为：
（和几个男人睡过；你多久没做爱了；恋爱经验多少，发生性关系了嘛；我 3 年没性生活了；之前睡过一个很软的妹子；你做爱喜欢什么姿势；你还是处女吗；我还是处男）
5.询问对方/阐述自己性能力/性癖的行为：
（你喜欢性欲强比较的男生么；你能做我的狗/主人吗；你多高多重什么cup啊；你胸多大；让你舒服，我是绿帽癖，下面18cm，看看胸/屁股）
6.聊骚/描述色情行为/暗示想发生性关系的行为
（那可以色色嘛；都脱掉那样打；分手是因为他下面不够大吗；想摸摸大腿根；想舔你的脚；你看着好骚；馋你身子；试婚/试爱；能接受炮友吗、素睡、晨勃了、来酒店找我吗；我年龄小，但我下面不小）
7.双方你情我愿聊骚时发送的黄图（男女露点、勃起内裤照、性行为、女性乳晕）
8.SM类（皮鞭、手铐）、润滑油、情趣内衣（未穿着状态）
9.你情我愿聊天发送低俗类图片召回自见

违规类型：重度色情
判断标准：1.发送裸露生殖器或明显性行为的图片/视频（男女生殖器、勃起内裤照、性行为）
2.提供性交易场所、性交易工作者、色情服务（如：原味、丝足、卖淫、援交、坐台、出台、有偿约、福利姬、前列腺按摩、男公关及其他）
3.进行招嫖、找小姐、招聘鸡鸭等行为（被动招嫖不管）
4.发布成人av电影、色情直播、情色片、三级片、偷拍走光露点相关的广告信息，包括视频、图片、链接、联系方式（索要网址给警告）
5.聊天中明确表示自己是约炮、找情人、包养的交友意图
（我就直说了，需要被包养吗；YP 吗；约酒店嘛，想和你睡；想找个长期固定py）
6.明确的性邀约行为（包括找 3p、换妻、单男、多人淫乱、人兽、乱伦的）
（3p考虑吗；文爱吗；2000过夜可以吗；裸聊吗；我只想找生理/肉体上的陪伴，你愿意吗）
7.聊天中存在冒犯性/挑逗性的文字或性行为描述：明确摸、舔、吸等行为&目标是胸、生殖器等敏感部位
（可以坐我脸上吗，想舔你下面；你下面的小洞能不能让咱们小家伙热乎热乎；我在打飞机）
8.自慰类（跳蛋、飞机杯）、刺激类（龟头冠状沟环、阴茎外套）、充气娃娃、避孕套等情趣用品图片
9.询问对方/阐述自己性器官的行为
（我JB很大你能接受么；你小穴水多不多；想试试你下面的洞紧不紧）
10.当前举报为色情警告基础上，用户在百条向≥3人发送构成色情警告处罚的会话，可直接升级ban处理（除当前举报外，无需区分是否你情我愿）
11.站外色情举报合理且为色情警告基础以上（除你情我愿聊崩）并且站内出现刷屏引流行为

违规类型：判罚标准
判断标准：双方你情我愿相互使用色情低俗语音、文字等内容进行交流不做处罚
女方先表明自己不找对象（或资料存在擦边低俗类内容）而引发男方询问能否约炮等骚扰行为被举报的不处罚
备注：需要明确识别为双方在相互撩骚，如果是单方面，则正常按照色情规则处置
已召回部分不再重复处罚

违规类型：轻度广告
判断标准：以曝光产品或服务为目的：
1.表明职业并曝光产品或服务带有营销信息
2.产品标识/名称+产品信息介绍
3.宣传话术/引导性话术
4.免费赠送产品、或诱导关注赠送产品等
5.诱导至其他app（单次发送警告处罚，向≥3人发送链接或引导下载可直接广告封号）
6.介绍产品功能同时有联系方式（包含支付宝、微信等及其他收款码类）
7.明确有拉客行为的用户，例如：购买联系我（普通商业广告）
8.明确为陪玩职业
备注：聊天过程中提及自己的身份为：保险、微商、销售、客服（销售类型）、培训班老师、房产中介、客户经理及其他带有销售性质的职业不管控

违规类型：严重广告
判断标准：有贩卖行为（除表明自己是购买方）：
1.明确酒吧、KTV、夜场夜店工作，出现拉客营销行为广告或确定为营销、ktv公主（不涉及钱色交易的）
2.明确招代理、招加盟
3.从事成人用品、两性保健品
4.平台+邀请码内容。明确写明是邀请码
5.网络兼职：刷单、打字、韵聊引流、任务引流及其他需要在家或通过线上完成的赚钱任务
6.减肥广告（描述自己减肥成功向其他人分享自己的减肥老师）
7.麻将托、酒托、饭托（酒托饭托向多个用户发送同一位置）、仙侠类游戏托（王者荣耀，吃鸡等常见手游不算）、情趣用品托（引导用户在指定售卖机中购买）、台球助教从业者；备注：带图举报 + 2条以上不同人的历史举报（必须是诈骗类型的举报标签）
8.竞品类出现拉客行为
9.明确来平台目的为拓展客户
10.陪玩拉客营销行为
11.台球助教或台球拉客营销
12.足疗、spa、按摩拉客行为

违规类型：引流
判断标准：话术类引流：
1.直接或间接导流至其他平台。例如：加了吗，加了告诉我/我心情不好加微信/失恋了加微信/遇到渣男了加微信/加了说一下/你多大了我们加qq吧（向多个账号（≥3）发送）
2.一个账号多次发送多个（两个及以上）同平台社交账号，例如：用户有多个微信号且聊天重复多次
3.主动向其他用户索要联系方式，但不透露自己的联系方式（刷屏比例达70%）

违规类型：引流
判断标准：非话术且社交账号类刷屏：
1.全社交账号或导流语言刷屏（90% 以上内容），如掺杂无意义类内容也算入【限女性，男性纯引流文本/图片/语音达到90%才可封禁】
2.向≥3人说自己是主播且有社交账号引流行为

违规类型：引流
判断标准：向多个账号（≥3）发送：
1.快手，抖音号等刷屏（补充话术是否所有其他社交平台账号）
2.助力、分红包其他合理平台，未涉及其他用户个人利益的链接地址。例如：拼多多邀请砍单、花小猪邀请砍单类
备注：如刷屏比例达90%可按非话术引流规则广告封号

违规类型：nan
判断标准：引流刷屏比例50%—90%

违规类型：轻度辱骂
判断标准：1.阴阳怪气、恶意调侃类词汇
例：高冷给谁看呢？；真下头、真晦气；X你妹、X个der（X不能为操、日等性行为动词）；哪来的优越感？；你不配；跟你有球关系？；不会说话就闭嘴；你真能装；不太喜欢河南的；你有一点矮；小老婆；小情人
2.对社会背景（工作、学校、学历、收入等）负面评价，例：你这么点收入还想养我？；就你这样也能当博士？；就这素质还老师呢？
3.非恶意针对外貌攻击，性别攻击例：你嘴长的有点奇怪；你眼睛有点小；你是男是女；你长的又不是特别好看；
4.非针对对方辱骂攻击，仅针对对方提出的问题进行贬低，例：你怎么会问这么弱智的问题？
5.小胖/小笨蛋/小丑鱼/小傻子等类似于亲密爱称召回不处罚，人工举报给提醒
6.滑了为什么不说话/不说话就别滑/不说话你匹配什么召回自见，人工举报不处罚
备注：当提醒类语句+脏话口头语，升级为警告处罚，例：你他妈配吗？；真他妈下头

违规类型：重度辱骂
判断标准：使用脏话、侮辱性、恶意低俗粗鄙的词汇，有指向性的诅咒、谩骂他人，如：你妈死了、死全家、给我滚远点（口头语及非针对举报人的除外）
备注：口头语，语气词——不处罚；包括但不限于：沃日、他妈的、傻逼等
1.叫嚣让对方去死或者患上严重疾病或残疾，例如：祝你死全家、神经病
2.性相关的贬低、辱骂，例如：装逼、傻逼、骚逼、不要逼脸、你必得艾滋
3.涉及动作+家人的贬低、诅咒，例如：你是狗娘养的、我是你爸爸、儿子真乖
4.对外貌、生理特征、年龄等负面评价，例如：你30长得像40岁一样；你个死胖子；你太丑了；你个老女人；你眼瞎啊？；你这身高和残疾没区别；
5.通过贬低他人无性生活实施攻击
6.诅咒他人在今后的情感上有缺失、严重诅咒他人例如：找不到对象、一辈子一个人、永远孤单寂寞、生孩子没屁眼、已读不回生不了小孩
备注：开玩笑或者合理场景除外，例如：那你真的打算这辈子不结婚了？
7.给他人发送引人不适的内容，例如恐怖视频、排泄物等
8. 对人的智商攻击，例如：智障、脑残、白痴、你脑子没事吧？
9. 把人和动物做对比，你是狗，你是猪
10.把人和污秽物、排泄物做对比，例如：你这个搅屎棍、你个垃圾
11.涉及地域攻击，例如：你果然是个农村佬；河南就是骗子多；

违规类型：重度辱骂
判断标准：1.宣称对方参与性行为的辱骂，例如：活该被操、干你下面
2.利用性相关身份去贬低他人，例如：婊子、妓女、你/你妈是出来卖B的吧
3.站外给对方发送辱骂轰炸短信等恶劣骚扰行为（如无法分辨哪一方先出现不友善行为，可先看站内举报方是否存在挑衅或先辱骂行为，若没有此类行为，可按站外证据直接处罚）
4.造他人黄谣，例如：我朋友跟你睡过

违规类型：重度辱骂
判断标准：频繁恶意辱骂：
百条信息中出现三次及三次以上对不同用户出现的辱骂或不友善行为（无差别攻击，对应三个及以上不同接收id）

违规类型：判罚标准
判断标准：单方面辱骂：人工举报-按尺度最严重那条处罚，召回举报——辱骂内容如属于提醒处罚范围内按自见处理，其他级别不变
双方互骂：人工举报——按照此段最严重的标签处罚，召回举报——自见
已召回部分不再重复处罚

违规类型：其他辱骂情况
判断标准：1.举报人发布骚扰内容导致被女性回击辱骂：
（1）亲密类称呼或试图建立关系，例如：宝贝、老婆、媳妇、宝宝、亲爱的、能做我老婆吗
备注：一些泛用类词汇除外：小姐姐、小仙女、小可爱、美女、亲等
（2）女性胸部、屁股、腿、脚部位正面评价或提及，例如：你胸多大
（3）发送冒犯类内容：来我被窝里睡吧、想看看对方敏感部位（非色情）、对找对象问题一再追问、晚上一起玩、能亲你吗
（4）发送有性暗示类内容或者试图把话题往色情方面引导（可能和上条有重合），例如：分手是因为他不够大吗；班有我好上吗
2.辱骂公司平台行为
"""

    user_template = """当前版本的用户聊天行为审核程序如下：
```python
{current_program}
```

这一版本程序是提升用户聊天内容审核准确性的一次尝试，在规划过程中，该尝试的目标和方案如下：
<intention>
{intention}
</intention>


现在，在这一版程序完成之后，为了得到能有助于进一步迭代程序的反馈信息，我们在一些真实存在的聊天数据上实际测试了该审核程序，同时也请人类审核员对同样的聊天内容提供了绝对正确的审核标准。人类审核员的评判结果是可信的，是我们开发自动化审核的标杆。这些结果如下所示：
{example_cases}


现在，请你评估一下这一版审核程序是否很好地落实了我们在<intention>中提及的具体解决方案，分析一下真实聊天数据上的测试结果，评价该解决方案能否解决<intention>中提出的问题，成功了还是失败了，为什么？并据此判断这条路线是否能提升模型性能的可行解决方案。
同时，通过对比不同用例中，当前版本程序的审核结果与人类审查员给出的可信结果，找出目前程序的问题，即为什么当前版本的程序结果仍然会和人类审查员给出的结果不一致。

在充分分析之后，你需要提供一份能有助于后续代码改进的反思报告。请将上述分析中，能有助于代码迭代优化的结论，以准确的语言写在 <reflection></reflection> 标签中。如果你认为有需要的话，也可以在reflection中引用具体的用例。反思报告应尽可能简洁，以便于阅读和理解。
"""


    current_program = re.sub(r'"""本次优化意图：.*?"""', '', RAW_PROGRAM, flags=re.DOTALL).strip()

    intention_match = re.search(r'"""本次优化意图：(.*?)"""', RAW_PROGRAM, re.DOTALL)
    if intention_match:
        raw_intention = intention_match.group(1).strip()
    else:
        raw_intention = "这是未经优化的最原始代码版本，直接调用一次大语言模型完成用户消息审核，是最简单的业务逻辑。"

    user_message = user_template.format(
        current_program = current_program,
        intention = raw_intention,
        example_cases = format_df(df_input, all_quick_overview),
    )

    client = AsyncOpenAI(
        base_url='https://llm-api.p1.cn',
        api_key=os.environ["OPENAI_API_KEY"],
    )
    
    async with sem:
        response = await client.chat.completions.create(
            model="openrouter/openai/gpt-5-chat",
            messages=[
                {"role": "user", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            # max_tokens=8192,
            # top_p=1.0,
        )

    match = re.search(r"<reflection>\s*(.*?)\s*</reflection>", response.choices[0].message.content, re.DOTALL)
    if not match:
        raise Exception
    return match.group(1).strip()


def format_df(df: pd.DataFrame, all_quick_overviews, threshold: float=0.5):
    out_FP = []
    out_FN = []
    out_TN = []
    out_TP = []
    format = '''## 待审核消息: {message}

## 消息所在对话: \n{context}\n

## 当前程序审核的行为逻辑简述: 
{quick_overview}

## 当前程序得到的预测结果：{predicted}

## 人类审核员的审核结果（正确答案）: {ground_truth}

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
            ground_truth="违规 (1)" if row['label'] else "未违规 (0)",
            predicted="违规 (1)" if predicted else "未违规 (0)",
            quick_overview=all_quick_overviews[idx],
        )

        if predicted == 1 and row['label'] == 0:
            out_FP.append(out)
        elif predicted == 0 and row['label'] == 1:
            out_FN.append(out)
        elif predicted == 1 and row['label'] == 1:
            out_TP.append(out)
        elif predicted == 0 and row['label'] == 0:
            out_TN.append(out)

    sampled_FP = random.sample(out_FP, min(len(out_FP), 20))
    sampled_FP = [f"<false_positive_example_{i}>\n{item}\n</false_positive_example_{i}>" for i, item in enumerate(sampled_FP)]
    sampled_FN = random.sample(out_FN, min(len(out_FN), 20))
    sampled_FN = [f"<false_negative_example_{i}>\n{item}\n</false_negative_example_{i}>" for i, item in enumerate(sampled_FN)]
    sampled_TP = random.sample(out_TP, min(len(out_TP), 10))
    sampled_TP = [f"<true_positive_example_{i}>\n{item}\n</true_positive_example_{i}>" for i, item in enumerate(sampled_TP)]
    sampled_TN = random.sample(out_TN, min(len(out_TN), 10))
    sampled_TN = [f"<true_negative_example_{i}>\n{item}\n</true_negative_example_{i}>" for i, item in enumerate(sampled_TN)]

    return f'''共对比了{len(all_quick_overviews)}条数据上的审核结果，其中：
- 人类认为不违规但被程序错误预测为违规的(False Positive)(共计 {len(out_FP)} 条)
- 人类认为违规但被程序错误预测为不违规的(False Negative)(共计 {len(out_FN)} 条)
- 人类认为不违规且被程序正确预测为不违规的(True Negative)(共计 {len(out_TN)} 条)
- 人类认为违规且被程序正确预测为违规的(True Positive)(共计 {len(out_TP)} 条)

以下展示部分用例：
{'\n\n'.join(sampled_FP)}

{'\n\n'.join(sampled_FN)}

{'\n\n'.join(sampled_TN)}

{'\n\n'.join(sampled_TP)}
    '''


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
            'precision': precision,
            'recall': recall,
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
    all_quick_overviews: List = [None] * len(batches)

    early_stopped = False
    finished = 0

    cancel_unfinished = lambda: [t.cancel() for t in tasks if not t.done()]    

    for fut in asyncio.as_completed(tasks):
        idx, df_chunk, quick_overview = await fut
        finished += 1

        all_results[idx] = df_chunk
        all_quick_overviews[idx] = quick_overview

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

        feedback_examples = format_df(res_df, all_quick_overviews)
        feedback_reflection = await get_reflection(res_df, all_quick_overviews)
        return f"""<feedback_examples>
{feedback_examples}
</feedback_examples>
<feedback_reflection>
{feedback_reflection}
</feedback_reflection>"""

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
    global RAW_PROGRAM
    RAW_PROGRAM = program
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
