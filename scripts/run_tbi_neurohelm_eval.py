#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：通过 API易（https://api.apiyi.com/v1）批量调用 9 个 LLM，
      完成你的 TBI-NeuroHELM / 脑外伤神经内科测评 benchmark。

默认要求：把下面几个文件和本脚本放在同一个文件夹：
  - 输入题集_英文版.csv
  - 案例编号-实例编号表.xlsx
  - 项目6_benchmark_12benchmark_240例_选择式过半_v7.xlsx
  - prompt.txt

最重要的地方：
  1. 在【用户需要修改的参数区】里填写每个模型对应的 API key。
  2. 在同一区域修改 WORKERS 就可以调节线程数。
  3. 正常直接运行：python run_tbi_medhelm_eval_CN_12keys_checked_v18_final_locked.py

输出文件夹：tbi_neurohelm_outputs
主要输出表：
  00_case_id_mapping_enriched.csv      匿名编号和真实 instance_id 的映射增强表
  01_model_outputs_raw.csv             9 个模型对 240 题的原始回答
  02_closed_item_scores.csv            封闭题 exact match 逐题判分
  03_open_jury_scores_raw.csv          开放题三模型 jury 原始评分
  04_open_item_scores.csv              开放题聚合后逐题评分
  05_item_scores_all.csv               封闭题 + 开放题合并逐题评分
  06_benchmark_scores_long.csv         模型 × benchmark 长表
  07_leaderboard.csv                   生成 Fig.3、Fig.4、Table 1 的核心表
  08_costs.csv                         生成 Fig.5 成本-性能图的核心表
  08b_costs_by_api_key_label.csv        按 12 个 key 用途标签汇总的核账表
  09_run_manifest.csv                  本次运行设置记录
  10_model_registry.csv                模型登记表
  11_error_retry_log.csv               报错和重试记录
  12_quality_control_report.csv        质量控制表：漏跑、解析失败、jury缺失检查
  13_category_scores.csv               模型 × 一级类别汇总分数
  14_subcategory_scores.csv            模型 × 子类别汇总分数
  15_question_type_scores.csv          封闭题/开放题分层表现
  16_pairwise_win_matrix.csv           模型两两比较胜率矩阵
  17_pairwise_win_long.csv             模型两两比较长表
  18_open_score_dimensions.csv         开放题各评分维度汇总
  19_jury_model_scoring_tendency_summary.csv 不同 jury 模型评分倾向和解析率
  20_jury_combination_robustness.csv   三评分模型的 7 种组合敏感性分析
  21_analysis_table_index.csv          图表/正文数据来源索引
  22_model_cost_performance_summary.csv Fig.5 和正文成本-性能汇总表
  23_jury_prompt_and_rubric.txt        Supplementary Fig.1 / Methods 用的 jury prompt 与评分锚点
  24_instance_distribution_summary.csv Supplementary Table 2 用的实例分布汇总表
  25_api_smoke_test_report.csv         smoke 模式真实 API 连通性测试报告
  26_paper_figure_table_source_map.csv 既定 5Fig+1Table+Extended/Supplementary 的数据来源映射
  27_paper_readiness_report.csv        正式写作前的完整性检查报告
  00_evaluation_config.json            本次评测配置指纹，防止不同配置混跑
  00_preflight_validation.csv          运行前输入检查结果（preflight 模式或正常运行均可生成）
  00_benchmark_manifest.csv            12 个 benchmark 的实例、题型和类别清单
  TBI_NeuroHELM_results.xlsx          汇总 Excel 工作簿

注意：
  - API key 不要发给别人，也不要上传到公开仓库。
  - 推荐为 9 个被测模型 + 3 个开放题评分模型分别创建 key，方便后台按 key 核算费用。
  - 如果你暂时只想用一个总 key，也可以填写 GLOBAL_APIYI_API_KEY 或系统环境变量 APIYI_API_KEY。
  - 脚本支持断点续跑：已经成功跑过的 runtime_id + model_name / judge 组合会自动跳过。
  - v18 会写入 00_evaluation_config.json；如果后续更改题库、prompt、模型列表、温度、token 上限或评分规则，请换新的 OUT_DIR，避免混跑。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


SCRIPT_VERSION = "v18_final_locked_paper_ready_20260529"


# ============================================================
# 用户需要修改的参数区
# ============================================================

# 1）API易 Key：
#    你现在准备建 12 个 key，所以建议在下面两个字典里逐个填写。
#    好处：API易后台可以按不同 key 查看消耗，更容易知道每个模型/评审模型花了多少钱。
#
#    注意：不要把填写了真实 key 的脚本发给别人，也不要上传到公开仓库。
#
#    如果你暂时不想填 12 个 key，也可以只填 GLOBAL_APIYI_API_KEY；
#    脚本会在某个模型专属 key 为空时，自动使用 GLOBAL_APIYI_API_KEY 兜底。
GLOBAL_APIYI_API_KEY = os.getenv("APIYI_API_KEY", "")

# 9 个被测模型的 key：用于跑 240 道 benchmark。
# 建议你在 API易后台创建 9 个被测模型 key，并按 test/模型名 命名，方便核账。
# 模型选择说明：移除 GPT-5/o3 这一类更容易出现参数兼容差异的 OpenAI reasoning/新旗舰模型；
# 用参考文献中也使用过、且你的 API易模型清单里存在的 GPT-4o、GPT-4o-mini、DeepSeek-R1 补位。
TEST_MODEL_API_KEYS = {
    "gpt-4o": os.getenv("APIYI_API_KEY_GPT_4O", ""),
    "gpt-4o-mini": os.getenv("APIYI_API_KEY_GPT_4O_MINI", ""),
    "deepseek-r1": os.getenv("APIYI_API_KEY_DEEPSEEK_R1", ""),
    "claude-opus-4-7": os.getenv("APIYI_API_KEY_CLAUDE_OPUS_4_7", ""),
    "claude-sonnet-4-6": os.getenv("APIYI_API_KEY_CLAUDE_SONNET_4_6", ""),
    "gemini-3.1-pro-preview": os.getenv("APIYI_API_KEY_GEMINI_3_1_PRO_PREVIEW", ""),
    "gemini-3-flash-preview": os.getenv("APIYI_API_KEY_GEMINI_3_FLASH_PREVIEW", ""),
    "deepseek-v3.2": os.getenv("APIYI_API_KEY_DEEPSEEK_V3_2", ""),
    "qwen3.7-max": os.getenv("APIYI_API_KEY_QWEN3_7_MAX", ""),
}

# 3 个开放题评分 LLM-jury 的 key：用于给 open_task 打分。
# 为了更贴近参考文献的设计，3 个评分模型从 9 个被测模型中挑选，并尽量覆盖不同厂商。
# 注意：gpt-4o、claude-opus-4-7、gemini-3.1-pro-preview 同时也是被测模型，
# 但这里仍然建议单独建 judge key，这样后台能区分“作答成本”和“评分成本”。
JUDGE_MODEL_API_KEYS = {
    "gpt-4o": os.getenv("APIYI_API_KEY_JUDGE_GPT_4O", os.getenv("APIYI_API_KEY_GPT_4O", "")),
    "claude-opus-4-7": os.getenv("APIYI_API_KEY_JUDGE_CLAUDE_OPUS_4_7", os.getenv("APIYI_API_KEY_CLAUDE_OPUS_4_7", "")),
    "gemini-3.1-pro-preview": os.getenv("APIYI_API_KEY_JUDGE_GEMINI_3_1_PRO_PREVIEW", os.getenv("APIYI_API_KEY_GEMINI_3_1_PRO_PREVIEW", "")),
}

# 2）线程数：
#    线程越大跑得越快，但越容易触发限流或超时。
#    建议先用 5 或 10 测试；确认稳定后再改成 20、30、50。
WORKERS = 5

# 3）运行模式：
#    "all"       = 从模型作答、客观题判分、开放题 jury、汇总表全部运行
#    "infer"     = 只跑 9 个被测模型作答
#    "closed"    = 只给封闭题判分
#    "jury"      = 只跑开放题 LLM-jury 评分
#    "aggregate" = 只重新汇总已有结果表，适合改价格表、胜率口径或论文汇总表后重算
#    "smoke"     = 小规模真实 API 连通性测试：每个被测模型各跑 1 道封闭题和 1 道开放题，3 个 judge 各评 1 次
#    "preflight" = 只检查输入文件、编号映射、题型数量和预期调用量，不调用 API
MODE = "preflight"  # public default: validate inputs without calling APIs

# 4）API地址：API易 OpenAI-compatible 接口。
API_BASE = "https://api.apiyi.com/v1"

# 5）输入文件名：通常不用改，除非你的文件改名了。
INPUT_CSV = "输入题集_英文版.csv"
MAPPING_XLSX = "案例编号-实例编号表.xlsx"
BENCHMARK_XLSX = "项目6_benchmark_12benchmark_240例_选择式过半_v7.xlsx"
PROMPT_TXT = "prompt.txt"
OUT_DIR = "tbi_neurohelm_outputs"

# 6）生成长度：
#    封闭题只需要输出选项和简短依据，所以短一点。
#    开放题和 jury 需要更长输出。
CLOSED_MAX_TOKENS = 3072
OPEN_MAX_TOKENS = 6144
JURY_MAX_TOKENS = 4096

# 7）温度：评测建议为 0，保证尽量稳定可复现。
TEMPERATURE = 0.0

# 8）API 参数策略：
#    False = 严格评测模式。所有模型统一发送 temperature 和 max_tokens；
#            如果某模型不接受这些参数，直接记录错误，不静默改参数。
#    True  = 兼容模式。仅在接口明确返回参数错误时，自动尝试 max_completion_tokens 或去掉问题参数。
#    当前模型方案已移除 GPT-5/o3 这类参数口径最容易不同的 OpenAI reasoning/新旗舰模型，
#    因此默认采用严格模式，以保证审稿时“同一请求参数”更容易解释。
ALLOW_API_PARAM_FALLBACK = False

# 9）超时和重试：网络不稳定或模型慢时可适当调大。
TIMEOUT = 180
MAX_RETRIES = 5

# 10）是否强制重跑：
#    False = 断点续跑，跳过已经跑过的结果。
#    True  = 不跳过，会重新请求 API；不建议随便打开，容易重复花钱。
FORCE_RERUN = False

# 11）价格表：
#     DEFAULT_PRICES_USD_PER_1M 中已放入一套可运行的默认价格，用于先跑通和估算成本。
#     其中部分模型沿用你之前从 API易页面核对过的价格；gpt-4o、gpt-4o-mini、
#     deepseek-r1 已根据你提供的 API易截图更新提示价和补全价。
#     若正式运行时 API易价格再次变化，填入外部 CSV 后用 MODE="aggregate" 重新汇总即可。
#     第一次运行仍会自动生成 tbi_neurohelm_outputs/model_prices_template.csv 方便你检查/备份。
PRICE_FILE = ""

# ----------------------------
# 固定的模型方案：9 个被测模型 + 3 个开放题评审模型
# ----------------------------
TEST_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "deepseek-r1",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "deepseek-v3.2",
    "qwen3.7-max",
]

JUDGE_MODELS = [
    "gpt-4o",
    "claude-opus-4-7",
    "gemini-3.1-pro-preview",
]

MODEL_PROVIDER = {
    "gpt-4o": "OpenAI",
    "gpt-4o-mini": "OpenAI",
    "deepseek-r1": "DeepSeek",
    "claude-opus-4-7": "Anthropic",
    "claude-sonnet-4-6": "Anthropic",
    "gemini-3.1-pro-preview": "Google",
    "gemini-3-flash-preview": "Google",
    "deepseek-v3.2": "DeepSeek",
    "qwen3.7-max": "Alibaba/Qwen",
}

# 下面已经填入你从 API易模型定价页面截图确认的价格。
# 单位：每 100 万 tokens 的美元价格。
# input = 提示价格；output = 补全价格。
# 正常情况下不需要再手动填写价格表。
DEFAULT_PRICES_USD_PER_1M = {
    # 单位均为 USD / 1M tokens。
    # 模型替换和价格说明：
    #   - 原 GPT-5.5      位置改为 gpt-4o
    #   - 原 GPT-5.4-mini 位置改为 gpt-4o-mini
    #   - 原 o3           位置改为 deepseek-r1
    # gpt-4o / gpt-4o-mini / deepseek-r1 已按用户提供的 API易截图更新：
    #   gpt-4o:       提示价 2.5000，补全价 10.0000 USD / 1M tokens
    #   gpt-4o-mini:  提示价 0.1500，补全价 0.6000 USD / 1M tokens
    #   deepseek-r1:  提示价 0.5700，补全价 2.2800 USD / 1M tokens
    "gpt-4o": {"input": 2.500, "output": 10.000},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "deepseek-r1": {"input": 0.570, "output": 2.280},
    "claude-opus-4-7": {"input": 5.000, "output": 25.000},
    "claude-sonnet-4-6": {"input": 3.000, "output": 15.000},
    "gemini-3.1-pro-preview": {"input": 1.800, "output": 10.800},
    "gemini-3-flash-preview": {"input": 0.440, "output": 2.640},
    "deepseek-v3.2": {"input": 0.280, "output": 0.420},
    "qwen3.7-max": {"input": 1.714, "output": 5.142},
}


def get_api_key_for_model(phase: str, model: str, fallback_key: str = "") -> str:
    """根据调用阶段和模型名取 key。

    phase = "test"  表示被测模型作答，优先用 TEST_MODEL_API_KEYS。
    phase = "judge" 表示开放题评分，优先用 JUDGE_MODEL_API_KEYS。
    如果对应专属 key 没填，就用 fallback_key 或环境变量 APIYI_API_KEY 兜底。
    """
    if phase == "test":
        key = TEST_MODEL_API_KEYS.get(model, "")
    elif phase == "judge":
        key = JUDGE_MODEL_API_KEYS.get(model, "")
    else:
        key = ""
    return (key or fallback_key or os.getenv("APIYI_API_KEY", "")).strip()


def key_label_for_model(phase: str, model: str) -> str:
    """只记录 key 的用途标签，不保存真实 key，避免结果表泄露密钥。"""
    return f"{phase}/{model}"


def build_api_clients(api_base: str, timeout: int, max_retries: int, fallback_key: str = "") -> Dict[Tuple[str, str], APIYiClient]:
    """为每个模型/阶段创建独立 client。

    这样即使同一个模型同时作为被测模型和评分模型，也可以使用不同 key，
    方便你在 API易后台分别查看作答成本和评分成本。
    """
    clients: Dict[Tuple[str, str], APIYiClient] = {}
    missing = []
    for m in TEST_MODELS:
        key = get_api_key_for_model("test", m, fallback_key=fallback_key)
        if not key:
            missing.append(f"test/{m}")
        else:
            clients[("test", m)] = APIYiClient(api_key=key, base_url=api_base, timeout=timeout, max_retries=max_retries)
    for m in JUDGE_MODELS:
        key = get_api_key_for_model("judge", m, fallback_key=fallback_key)
        if not key:
            missing.append(f"judge/{m}")
        else:
            clients[("judge", m)] = APIYiClient(api_key=key, base_url=api_base, timeout=timeout, max_retries=max_retries)
    if missing:
        raise RuntimeError(
            "以下模型/阶段没有填写 API key：" + ", ".join(missing) +
            "。请在脚本顶部 TEST_MODEL_API_KEYS / JUDGE_MODEL_API_KEYS 中填写，"
            "或者填写 GLOBAL_APIYI_API_KEY 作为兜底。"
        )
    return clients

CATEGORY_MAP_BY_PREFIX = {
    "CDS": "Clinical Decision Support",
    "DOC": "Clinical Documentation Generation",
    "COM": "Patient Communication and Education",
    "RES": "Medical Research Assistance",
}

# ----------------------------
# 工具函数：时间、路径、读写文件、tokens 粗略估计等
# ----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    """把秒数格式化成适合控制台进度显示的 h/m/s。"""
    seconds = max(0, int(round(seconds)))
    hh, rem = divmod(seconds, 3600)
    mm, ss = divmod(rem, 60)
    if hh:
        return f"{hh:d}h{mm:02d}m{ss:02d}s"
    if mm:
        return f"{mm:d}m{ss:02d}s"
    return f"{ss:d}s"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """计算输入文件 SHA256。用于 run manifest 和配置指纹，便于论文复现。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_dumps(obj: Any) -> str:
    """稳定 JSON 序列化，保证同一配置得到同一 hash。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def build_evaluation_config_payload(
    system_prompt: str,
    args: argparse.Namespace,
    input_csv: Path,
    mapping_xlsx: Path,
    benchmark_xlsx: Path,
    prompt_txt: Path,
) -> Dict[str, Any]:
    """构建评测配置指纹。

    这个 hash 不用于安全加密，而是用于防止不同题库、prompt、模型、温度、token budget
    或评分规则的结果混在同一个输出目录中。
    """
    return {
        "script_family": "TBI-NeuroHELM MedHELM-style evaluation",
        "test_models": TEST_MODELS,
        "judge_models": JUDGE_MODELS,
        "model_provider": MODEL_PROVIDER,
        "temperature": args.temperature,
        "closed_max_tokens": args.closed_max_tokens,
        "open_max_tokens": args.open_max_tokens,
        "jury_max_tokens": args.jury_max_tokens,
        "allow_api_param_fallback": ALLOW_API_PARAM_FALLBACK,
        "api_param_policy": "strict unified temperature and task-type max_tokens",
        "input_csv_sha256": sha256_file(input_csv),
        "mapping_xlsx_sha256": sha256_file(mapping_xlsx),
        "benchmark_xlsx_sha256": sha256_file(benchmark_xlsx),
        "prompt_txt_sha256": sha256_file(prompt_txt),
        "system_prompt_sha256": sha256_text(system_prompt),
        "jury_system_prompt_sha256": sha256_text(JURY_SYSTEM_PROMPT),
        "open_task_scoring_anchors_sha256": sha256_text(OPEN_TASK_SCORING_ANCHORS),
        "closed_scoring_rule": "exact_match_single_choice",
        "open_primary_score": "mean(accuracy_score, completeness_score, clarity_score), normalized as (raw-1)/4",
        "win_rate_policy": "primary strict win: score > opponent; tie=0.5 and tie=win sensitivity columns saved",
    }


def write_and_validate_evaluation_config(out_dir: Path, payload: Dict[str, Any], mode: str, force: bool) -> str:
    """写入 00_evaluation_config.json，并阻止不同配置在同一 OUT_DIR 混跑。

    如需更改题库、prompt、模型、温度、token 上限或评分规则，推荐直接换新的 OUT_DIR。
    """
    cfg_path = out_dir / "00_evaluation_config.json"
    new_hash = sha256_text(stable_json_dumps(payload))
    existing_result_files = [
        "01_model_outputs_raw.csv",
        "02_closed_item_scores.csv",
        "03_open_jury_scores_raw.csv",
        "04_open_item_scores.csv",
        "05_item_scores_all.csv",
    ]
    if cfg_path.exists():
        try:
            old = json.loads(cfg_path.read_text(encoding="utf-8"))
            old_hash = str(old.get("evaluation_config_hash", ""))
        except Exception:
            old_hash = ""
        if old_hash and old_hash != new_hash and mode != "preflight" and not force:
            raise RuntimeError(
                "当前 OUT_DIR 中已存在不同评测配置的 00_evaluation_config.json。"
                "为避免不同 prompt/模型/参数/题库结果混跑，请更换 OUT_DIR，或在确认要重跑时使用 --force。"
            )
    elif any((out_dir / f).exists() and (out_dir / f).stat().st_size > 0 for f in existing_result_files) and mode != "preflight" and not force:
        raise RuntimeError(
            "当前 OUT_DIR 中已有正式结果表，但没有 v18 配置指纹文件。"
            "为避免把旧版本结果和 v18 结果混合，请更换 OUT_DIR，或在确认要重跑时使用 --force。"
        )

    payload_to_write = dict(payload)
    payload_to_write["evaluation_config_hash"] = new_hash
    payload_to_write["script_version_recorded_at_config_write"] = SCRIPT_VERSION
    payload_to_write["config_written_at"] = now_iso()
    cfg_path.write_text(json.dumps(payload_to_write, ensure_ascii=False, indent=2), encoding="utf-8")
    return new_hash


def resolve_path(path_str: str, script_dir: Path) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    p2 = script_dir / path_str
    if p2.exists():
        return p2
    p3 = Path("/mnt/data") / path_str
    if p3.exists():
        return p3
    return p


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path) -> pd.DataFrame:
    # utf-8-sig 可以兼容 Excel 导出的 CSV，避免中文乱码。
    return pd.read_csv(path, encoding="utf-8-sig")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def append_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str], lock: threading.Lock) -> None:
    with lock:
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def load_existing_key_set(path: Path, key_cols: List[str]) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except Exception:
        return set()
    if df.empty or not all(c in df.columns for c in key_cols):
        return set()
    return set(tuple(str(row[c]) for c in key_cols) for _, row in df.iterrows())


def truthy_series(s: pd.Series) -> pd.Series:
    """把 CSV 中可能保存成 True/true/1/yes 的列统一转成布尔序列。"""
    return s.fillna("").astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def latest_rows_by_key(df: pd.DataFrame, key_cols: List[str], success_mask: Optional[pd.Series] = None) -> pd.DataFrame:
    """按 key 保留最后一条可用于分析的记录；优先保留成功记录。"""
    if df.empty or not all(c in df.columns for c in key_cols):
        return df.copy()
    tmp = df.copy()
    tmp["_row_order"] = range(len(tmp))
    if success_mask is None:
        tmp["_success_rank"] = 1
    else:
        tmp["_success_rank"] = success_mask.reindex(tmp.index).fillna(False).astype(int)
    sort_cols = [c for c in key_cols if c in tmp.columns] + ["_success_rank", "_row_order"]
    tmp = tmp.sort_values(sort_cols)
    tmp = tmp.drop_duplicates(key_cols, keep="last")
    return tmp.drop(columns=["_row_order", "_success_rank"], errors="ignore")


def inference_success_mask(raw: pd.DataFrame) -> pd.Series:
    if raw.empty:
        return pd.Series(dtype=bool)
    err_ok = raw.get("error_message", "").fillna("").astype(str).str.strip().eq("")
    response_ok = raw.get("raw_response", "").fillna("").astype(str).str.strip().ne("")
    return err_ok & response_ok


def jury_success_mask(jury: pd.DataFrame) -> pd.Series:
    if jury.empty:
        return pd.Series(dtype=bool)
    parse_ok = truthy_series(jury.get("judge_parse_success", pd.Series(False, index=jury.index)))
    score_cols = [c for c in ["accuracy_score", "completeness_score", "clarity_score", "overall_score"] if c in jury.columns]
    if score_cols:
        score_ok = jury[score_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    else:
        score_ok = pd.Series(False, index=jury.index)
    return parse_ok & score_ok


def dedupe_inference_raw(raw: pd.DataFrame) -> pd.DataFrame:
    return latest_rows_by_key(raw, ["runtime_id", "model_name"], inference_success_mask(raw) if not raw.empty else None)


def dedupe_jury_raw(jury: pd.DataFrame) -> pd.DataFrame:
    return latest_rows_by_key(jury, ["runtime_id", "model_under_eval", "judge_model"], jury_success_mask(jury) if not jury.empty else None)


def load_successful_inference_key_set(path: Path) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except Exception:
        return set()
    if raw.empty or not {"runtime_id", "model_name"}.issubset(raw.columns):
        return set()
    raw = raw.loc[inference_success_mask(raw)]
    return set(tuple(str(row[c]) for c in ["runtime_id", "model_name"]) for _, row in raw.iterrows())


def load_successful_jury_key_set(path: Path) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        jury = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except Exception:
        return set()
    required = {"runtime_id", "model_under_eval", "judge_model"}
    if jury.empty or not required.issubset(jury.columns):
        return set()
    jury = jury.loc[jury_success_mask(jury)]
    return set(tuple(str(row[c]) for c in ["runtime_id", "model_under_eval", "judge_model"]) for _, row in jury.iterrows())


def coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def approx_tokens(text: str) -> int:
    # 只有在 API 没有返回 usage 时才用这个粗略估算。
    if not text:
        return 0
    # 英文/中文混合文本，用字符数做一个保守估计。
    return max(1, math.ceil(len(text) / 4))

# ----------------------------
# 数据准备：读取输入题集、匿名编号映射表、benchmark 母表
# ----------------------------

def extract_benchmark_id(instance_id: str) -> str:
    # 例子：TBI-CDS-B01-I001 -> CDS-B01
    m = re.search(r"TBI-([A-Z]+-B\d+)-I\d+", str(instance_id))
    return m.group(1) if m else "UNKNOWN"


def prefix_from_benchmark(benchmark_id: str) -> str:
    return str(benchmark_id).split("-")[0] if benchmark_id and "-" in str(benchmark_id) else "UNKNOWN"


def load_and_prepare_data(input_csv: Path, mapping_xlsx: Path, benchmark_xlsx: Path, out_dir: Path) -> pd.DataFrame:
    input_df = safe_read_csv(input_csv)
    required = {"runtime_id", "question_type", "context", "prompt"}
    missing = required - set(input_df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    mapping_df = pd.read_excel(mapping_xlsx, dtype=str)
    if not {"runtime_id", "instance_id"}.issubset(mapping_df.columns):
        raise ValueError("Mapping xlsx must contain runtime_id and instance_id columns.")

    instances = pd.read_excel(benchmark_xlsx, sheet_name="instances", dtype=str)
    if "instance_id" not in instances.columns:
        raise ValueError("Benchmark xlsx sheet 'instances' must contain instance_id.")

    # 保留后续评分和聚合需要用到的重要元数据。
    keep_cols = [
        "instance_id", "benchmark_group_id", "benchmark_group_name", "benchmark_group_domain",
        "benchmark_group_subcategories", "subcategory", "instance_index_within_group", "task_form",
        "complexity", "reference_answer_or_key", "scoring_primary", "scoring_secondary",
        "scoring_scale", "score_fields", "detailed_scoring_rubric", "judge_prompt_core",
        "output_template_type",
    ]
    keep_cols = [c for c in keep_cols if c in instances.columns]
    meta = instances[keep_cols].copy()

    df = input_df.merge(mapping_df, on="runtime_id", how="left", validate="one_to_one")
    df = df.merge(meta, on="instance_id", how="left", validate="many_to_one")
    df["benchmark_id"] = df["benchmark_group_id"].fillna(df["instance_id"].map(extract_benchmark_id))
    df["category_prefix"] = df["benchmark_id"].map(prefix_from_benchmark)
    df["category"] = df["category_prefix"].map(CATEGORY_MAP_BY_PREFIX).fillna(df.get("benchmark_group_domain", "UNKNOWN"))
    df["benchmark_display_name"] = df["benchmark_id"].astype(str) + " - " + df.get("benchmark_group_name", "").fillna("").astype(str)

    # 保存增强版映射表，方便后续复现和检查。
    mapping_enriched = df[[
        "runtime_id", "instance_id", "benchmark_id", "benchmark_display_name", "category", "subcategory",
        "question_type", "complexity", "task_form", "reference_answer_or_key", "detailed_scoring_rubric",
    ]].copy()
    mapping_enriched.to_csv(out_dir / "00_case_id_mapping_enriched.csv", index=False, encoding="utf-8-sig")

    if df["instance_id"].isna().any():
        bad = df.loc[df["instance_id"].isna(), "runtime_id"].tolist()[:10]
        raise ValueError(f"Some runtime_id values were not found in mapping table, examples: {bad}")

    return df

# ----------------------------
# API易调用：OpenAI-compatible chat/completions 接口
# ----------------------------

class APIYiClient:
    def __init__(self, api_key: str, base_url: str, timeout: int = 120, max_retries: int = 5):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # requests.Session 不是严格线程安全对象；本脚本会多线程并发，
        # 所以为每个工作线程各自维护一个 session，避免连接池状态互相影响。
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(self.headers)
            self._thread_local.session = session
        return session

    @staticmethod
    def _normalize_content(content: Any) -> str:
        """兼容不同 OpenAI-compatible 代理可能返回的 message.content 格式。"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        parts.append(item["content"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()
        return str(content)

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = 1024,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        last_err: Optional[str] = None
        session = self._get_session()
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 400:
                    text = resp.text.lower()
                    if not ALLOW_API_PARAM_FALLBACK:
                        return {
                            "ok": False,
                            "content": "",
                            "raw_json": None,
                            "response_model": "",
                            "finish_reason": "",
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                            "error": f"HTTP 400: {resp.text[:500]}",
                        }
                    # 兼容模式：部分推理模型/代理模型可能不接受 temperature、max_tokens，
                    # 或要求使用 max_completion_tokens；遇到明确参数错误时自动调整后重试。
                    changed = False
                    if "temperature" in text and "temperature" in payload:
                        payload.pop("temperature", None)
                        changed = True
                    if "max_completion_tokens" in text and "max_tokens" in payload:
                        payload["max_completion_tokens"] = payload.pop("max_tokens")
                        changed = True
                    elif ("max_tokens" in text or "max token" in text) and "max_tokens" in payload:
                        payload.pop("max_tokens", None)
                        changed = True
                    if changed:
                        resp = session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    time.sleep(min(60, 2 ** attempt + random.random()))
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = ""
                finish_reason = ""
                if data.get("choices"):
                    choice0 = data["choices"][0]
                    msg = choice0.get("message") or {}
                    content = self._normalize_content(msg.get("content"))
                    finish_reason = choice0.get("finish_reason", "")
                usage = data.get("usage") or {}
                return {
                    "ok": True,
                    "content": content,
                    "raw_json": data,
                    "response_model": data.get("model", ""),
                    "finish_reason": finish_reason,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "error": "",
                }
            except Exception as e:
                last_err = repr(e)
                time.sleep(min(60, 2 ** attempt + random.random()))
        return {"ok": False, "content": "", "raw_json": None, "response_model": "", "finish_reason": "", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "error": last_err or "unknown error"}


def build_test_messages(system_prompt: str, row: pd.Series) -> List[Dict[str, str]]:
    user_content = (
        f"Case ID: {row['runtime_id']}\n"
        f"question_type: {row['question_type']}\n\n"
        f"context:\n{row['context']}\n\n"
        f"prompt:\n{row['prompt']}\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

# ----------------------------
# 封闭题答案解析和 exact match 判分
# ----------------------------

def option_letters_from_prompt(prompt: str) -> List[str]:
    letters = re.findall(r"(?:^|[\s;；。\.])([A-H])\s*[\.．、:]", str(prompt))
    if not letters:
        letters = re.findall(r"\b([A-H])\b", str(prompt))
    out = []
    for x in letters:
        if x not in out:
            out.append(x)
    return out or list("ABCDEFGH")


def normalize_letters(letters: Iterable[str]) -> str:
    uniq = sorted({x.upper() for x in letters if isinstance(x, str) and re.fullmatch(r"[A-H]", x.upper())})
    return ",".join(uniq)


def extract_answer_letters(text: Any, valid_options: Optional[List[str]] = None, reference_mode: bool = False) -> str:
    """解析单选题答案，只返回一个选项字母。

    本 benchmark 的 closed_qa 全部为单选题，因此这里不再支持多选集合解析。
    解析顺序：先找明确标签，如 Final answer: A；再只看回答首行兜底，
    避免把 rationale 中的 GCS、TBI 或普通英文大写字母误识别为答案。
    """
    s = "" if text is None else str(text)
    valid = set(valid_options or list("ABCDEFGH"))
    answer_letter = r"([A-H])"
    # 匹配后面要求是常见分隔符、空白或文本结束，避免把单词里的 A-H 抓出来。
    tail = r"(?=\s*(?:[>\.．、,，;；\)\]：:]|$|\n|\r))"
    patterns = [
        rf"正确答案\s*[:：]?\s*<?\s*{answer_letter}{tail}",
        rf"参考答案\s*[:：]?\s*<?\s*{answer_letter}{tail}",
        rf"Correct\s+answer\s*[:：]?\s*<?\s*{answer_letter}{tail}",
        rf"Final\s+answer\s*[:：]?\s*<?\s*{answer_letter}{tail}",
        rf"答案\s*[:：]?\s*<?\s*{answer_letter}{tail}",
        rf"选项\s*[:：]?\s*<?\s*{answer_letter}{tail}",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in valid:
                return letter

    # 兜底解析：只看模型回答开头，避免后面解释里的 GCS/TBI 等字母干扰。
    head = s.strip().splitlines()[0][:120] if s.strip() else ""
    m = re.match(r"^\s*<?\s*([A-H])\s*(?:[>\.．、,，;；\)\]\s]|$)", head, flags=re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    if reference_mode:
        # 标准答案字段的最后兜底：找第一个独立的选项字母。
        for x in re.findall(r"(?<![A-Z])([A-H])(?![A-Z])", s.upper()):
            if x in valid:
                return x
    else:
        # 模型回答的最后兜底：只在开头 120 字符内找独立选项字母。
        for x in re.findall(r"(?<![A-Z])([A-H])(?![A-Z])", head.upper()):
            if x in valid:
                return x
    return ""


def exact_match_score(parsed: str, reference: str) -> Tuple[int, bool]:
    if not parsed or not reference:
        return 0, False
    return (1 if parsed == reference else 0), True

# ----------------------------
# 开放题 LLM-jury 评分
# ----------------------------

JURY_SYSTEM_PROMPT = """You are an independent senior neurologist and medical AI evaluator. You evaluate traumatic brain injury neurology benchmark answers. You must be strict, fair, and base the score only on the provided case, task, reference answer or rubric, candidate response, and anchored scoring criteria. Do not infer unprovided clinical facts. Output JSON only."""


OPEN_TASK_SCORING_ANCHORS = """
Primary open-ended scoring follows a MedHELM-style three-dimension rubric. Score Accuracy, Completeness, and Clarity separately from 1 to 5. Safety and Overall are auxiliary dimensions and are not included in the primary leaderboard score.

Accuracy anchors:
5 = medically correct, faithful to the case and reference key points, no hallucinated facts, conclusions fully supported.
4 = main conclusion and most reasoning correct; only minor imprecision that would not change clinical interpretation.
3 = partly correct, but includes non-fatal over-interpretation, incomplete interpretation, or case misread that may affect reasoning.
2 = clear medical error, contradiction with case information, unsupported key claim, or wrong key conclusion.
1 = mostly wrong, severe hallucination, dangerous recommendation, or complete failure to address the task.

Completeness anchors:
5 = covers the core conclusion, key supporting evidence, uncertainty/boundaries, and necessary next-step recommendation or quality-control reminder.
4 = covers most required points; only minor details omitted.
3 = states the main conclusion but misses important evidence, risk factors, uncertainty, follow-up needs, or task-specific requirements.
2 = clearly incomplete; generic or superficial answer and omits multiple key elements.
1 = almost no task-relevant content or fails to address the requested clinical/research objective.

Clarity anchors:
5 = well organized, concise, readable, follows requested output structure, and directly usable for the intended task.
4 = mostly clear and usable; minor redundancy, wording, or formatting issues.
3 = understandable but weakly organized, priorities unclear, or requested structure partly followed.
2 = disorganized, verbose, ambiguous, or difficult to use for clinical/research decision-making.
1 = hard to understand, severely violates the requested format, or essentially off-task.

Auxiliary safety anchors:
5 = avoids unsafe advice, unsupported certainty, hallucinated management, and appropriately recommends escalation/deferral when needed.
4 = generally safe; minor wording could be more cautious but no clinically dangerous implication.
3 = some overconfidence, incomplete caution, or limited risk recognition, but no clearly dangerous advice.
2 = potentially unsafe advice or inappropriate certainty that could mislead decisions.
1 = dangerous recommendation, serious hallucination, or explicit advice likely to cause harm.

Cap and penalty rules:
- If the core conclusion directly contradicts the case/reference, Accuracy must be <= 2.
- If the response fabricates key history, examination, imaging, diagnosis, or treatment facts, Accuracy must be <= 3; if clinically risky, Accuracy must be <= 2.
- If central TBI risk elements are omitted when relevant, such as antithrombotic use, worsening consciousness, focal neurological deficit, seizure risk, imaging progression, or need for follow-up, Completeness must be <= 3.
- If the requested structure is not followed, Clarity must be <= 3; if the answer is off-task or not interpretable, Clarity must be <= 2.
- Reflect potential medical safety concerns in the auxiliary safety_score and rationale; no separate binary error labels are required.

Primary open-task normalized score downstream is calculated as:
score_norm = (mean(Accuracy, Completeness, Clarity) - 1) / 4 = (A + C + Cl - 3) / 12.
""".strip()


def build_jury_prompt(row: pd.Series, candidate_response: str) -> str:
    rubric = row.get("detailed_scoring_rubric") or row.get("judge_prompt_core") or "Evaluate medical accuracy, completeness, clarity, and safety."
    reference = row.get("reference_answer_or_key") or "No single gold answer; use the rubric and case information."
    return f"""
You are scoring one anonymized model response to an open-ended traumatic brain injury neurology benchmark item.

Case ID: {row['runtime_id']}
Question type: {row['question_type']}
Benchmark: {row.get('benchmark_id', '')}
Task form: {row.get('task_form', '')}

Context:
{row['context']}

Prompt/task:
{row['prompt']}

Reference answer or key points:
{reference}

Task-specific scoring rubric:
{rubric}

Global anchored scoring criteria:
{OPEN_TASK_SCORING_ANCHORS}

Candidate response to score:
{candidate_response}

Score each dimension from 1 to 5 using the anchored criteria above:
- accuracy_score: medical correctness, case fidelity, and consistency with the reference key points.
- completeness_score: coverage of clinically/research-important points required by the task.
- clarity_score: organization, readability, and fit to requested output format.
- safety_score: auxiliary score for unsafe advice, overclaiming, hallucinated facts, or inappropriate certainty.
- overall_score: auxiliary overall quality, 1 to 5.

Return JSON only with this exact schema:
{{
  "accuracy_score": <number 1-5>,
  "completeness_score": <number 1-5>,
  "clarity_score": <number 1-5>,
  "safety_score": <number 1-5>,
  "overall_score": <number 1-5>,
  "rationale": "brief reason in one or two sentences; mention any cap rule applied"
}}
""".strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    # 如果模型把 JSON 包在 ```json 代码块里，先去掉代码块。
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # 兜底：截取第一个 { 到最后一个 } 之间的内容尝试解析。
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except Exception:
            return None
    return None



def parse_jury_score_object(text: str) -> Tuple[Dict[str, Any], bool]:
    """解析 jury 输出。

    首选严格 JSON；如果评分模型偶尔输出了非 JSON 文本，使用标签+数字的兜底解析，
    尽量保住全量 jury 调用中的可用评分，并在 judge_parse_success 中记录解析是否成功。
    """
    js = extract_json_object(text)
    if isinstance(js, dict) and js:
        return js, True

    s = "" if text is None else str(text)
    label_map = {
        "accuracy_score": r"accuracy(?:_score)?|medical correctness",
        "completeness_score": r"completeness(?:_score)?",
        "clarity_score": r"clarity(?:_score)?",
        "safety_score": r"safety(?:_score)?",
        "overall_score": r"overall(?:_score)?",
    }
    recovered: Dict[str, Any] = {}
    for key, label_pat in label_map.items():
        m = re.search(rf"(?:{label_pat})\s*[:=\-]?\s*([1-5](?:\.0)?)", s, flags=re.IGNORECASE)
        if m:
            recovered[key] = m.group(1)
    rat = re.search(r"rationale\s*[:：]\s*(.+)", s, flags=re.IGNORECASE | re.DOTALL)
    if rat:
        recovered["rationale"] = rat.group(1).strip()[:1000]
    return recovered, bool(recovered)


def clamp_score(x: Any) -> Optional[float]:
    v = coerce_float(x)
    if v is None:
        return None
    return min(5.0, max(1.0, v))

# ----------------------------
# 价格和成本计算
# ----------------------------

def load_or_create_price_table(out_dir: Path, price_file: Optional[Path]) -> pd.DataFrame:
    template_path = out_dir / "model_prices_template.csv"
    rows = []
    for m in sorted(set(TEST_MODELS + JUDGE_MODELS)):
        p = DEFAULT_PRICES_USD_PER_1M.get(m, {"input": None, "output": None})
        rows.append({
            "model_name": m,
            "provider": MODEL_PROVIDER.get(m, ""),
            "input_price_per_1m": p["input"],
            "output_price_per_1m": p["output"],
            # 下面 3 列只用于阶梯计费模型。非阶梯模型留空。
            "tier_threshold_input_tokens": p.get("tier_threshold_input_tokens"),
            "input_price_over_threshold_per_1m": p.get("input_over_threshold"),
            "output_price_over_threshold_per_1m": p.get("output_over_threshold"),
            "notes": "脚本内置默认价格用于成本估算；gpt-4o/gpt-4o-mini/deepseek-r1 已按用户提供的 API易截图更新，正式论文成本表仍建议按 API易控制台实时价格复核，必要时用 PRICE_FILE 覆盖。",
        })
    default_df = pd.DataFrame(rows)
    if price_file and price_file.exists():
        user_prices = pd.read_csv(price_file, encoding="utf-8-sig")
        merged = default_df.drop(columns=["input_price_per_1m", "output_price_per_1m", "notes"], errors="ignore").merge(
            user_prices, on="model_name", how="left", suffixes=("", "_user")
        )
        return merged
    default_df.to_csv(template_path, index=False, encoding="utf-8-sig")
    return default_df


def model_cost(model: str, input_tokens: Any, output_tokens: Any, price_df: pd.DataFrame) -> float:
    """按 API易价格估算单次请求成本。

    普通模型：
        cost = input_tokens / 1e6 × input_price_per_1m
             + output_tokens / 1e6 × output_price_per_1m

    阶梯模型：
        如果 price_df 中存在 tier_threshold_input_tokens，并且本次请求的 input_tokens
        大于该阈值，则改用 over-threshold 的输入/输出价格。
        这里按 API易价格页的上下文区间理解为“输入 prompt tokens 阈值”。
    """
    inp = coerce_float(input_tokens) or 0.0
    out = coerce_float(output_tokens) or 0.0
    hit = price_df.loc[price_df["model_name"] == model]
    if hit.empty:
        return 0.0

    row = hit.iloc[0]
    in_price = coerce_float(row.get("input_price_per_1m")) or 0.0
    out_price = coerce_float(row.get("output_price_per_1m")) or 0.0

    threshold = coerce_float(row.get("tier_threshold_input_tokens"))
    if threshold is not None and inp > threshold:
        in_price = coerce_float(row.get("input_price_over_threshold_per_1m")) or in_price
        out_price = coerce_float(row.get("output_price_over_threshold_per_1m")) or out_price

    return inp / 1_000_000 * in_price + out / 1_000_000 * out_price

# ----------------------------
# 第 1 阶段：调用 9 个被测模型作答
# ----------------------------

RAW_OUTPUT_FIELDS = [
    "run_id", "runtime_id", "instance_id", "benchmark_id", "category", "subcategory",
    "question_type", "model_name", "model_snapshot", "api_response_model", "finish_reason", "provider", "api_key_label", "system_prompt_version",
    "context", "prompt", "raw_response", "parsed_answer", "parse_success", "error_message",
    "input_tokens", "output_tokens", "total_tokens", "usage_estimated", "latency_sec",
    "temperature", "max_tokens", "finished_at",
]

ERROR_FIELDS = ["run_id", "phase", "runtime_id", "instance_id", "model_name", "judge_model", "attempt", "error_type", "error_message", "finished_at"]


def run_inference(
    df: pd.DataFrame,
    system_prompt: str,
    clients: Dict[Tuple[str, str], APIYiClient],
    out_dir: Path,
    run_id: str,
    workers: int,
    force: bool,
    temperature: float,
    closed_max_tokens: int,
    open_max_tokens: int,
) -> None:
    out_path = out_dir / "01_model_outputs_raw.csv"
    err_path = out_dir / "11_error_retry_log.csv"
    lock = threading.Lock()
    err_lock = threading.Lock()
    done = set() if force else load_successful_inference_key_set(out_path)

    tasks = []
    for _, row in df.iterrows():
        for model in TEST_MODELS:
            key = (str(row["runtime_id"]), model)
            if key not in done:
                tasks.append((row, model))

    total_possible = len(df) * len(TEST_MODELS)
    skipped_successful = max(0, total_possible - len(tasks)) if not force else 0
    print(
        f"[模型作答] 阶段开始：待请求 {len(tasks)} 次；已成功跳过 {skipped_successful} 次；"
        f"理论总请求 {total_possible} 次；workers={workers}",
        flush=True,
    )
    if not tasks:
        return

    stage_started_at = time.time()
    progress_interval = 1 if len(tasks) <= 50 else 10

    def worker(job: Tuple[pd.Series, str]) -> Dict[str, Any]:
        row, model = job
        t0 = time.time()
        max_tokens = open_max_tokens if row["question_type"] == "open_task" else closed_max_tokens
        messages = build_test_messages(system_prompt, row)
        client = clients[("test", model)]
        res = client.chat_completion(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        latency = time.time() - t0
        content = res.get("content") or ""
        usage_estimated = False
        in_tok = res.get("prompt_tokens")
        out_tok = res.get("completion_tokens")
        tot_tok = res.get("total_tokens")
        if in_tok is None or out_tok is None:
            usage_estimated = True
            in_tok = approx_tokens(system_prompt + "\n" + messages[-1]["content"])
            out_tok = approx_tokens(content)
            tot_tok = in_tok + out_tok
        valid_options = option_letters_from_prompt(str(row.get("prompt", "")))
        parsed = ""
        parse_success = False
        if row["question_type"] == "closed_qa" and content:
            parsed = extract_answer_letters(content, valid_options=valid_options, reference_mode=False)
            parse_success = bool(parsed)
        elif row["question_type"] == "open_task" and content:
            parse_success = True
        err = "" if res.get("ok") else (res.get("error") or "unknown error")
        return {
            "run_id": run_id,
            "runtime_id": row["runtime_id"],
            "instance_id": row["instance_id"],
            "benchmark_id": row["benchmark_id"],
            "category": row["category"],
            "subcategory": row.get("subcategory", ""),
            "question_type": row["question_type"],
            "model_name": model,
            "model_snapshot": model,
            "api_response_model": res.get("response_model", ""),
            "finish_reason": res.get("finish_reason", ""),
            "provider": MODEL_PROVIDER.get(model, ""),
            "api_key_label": key_label_for_model("test", model),
            "system_prompt_version": sha256_text(system_prompt)[:12],
            "context": row["context"],
            "prompt": row["prompt"],
            "raw_response": content,
            "parsed_answer": parsed,
            "parse_success": parse_success,
            "error_message": err,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot_tok,
            "usage_estimated": usage_estimated,
            "latency_sec": round(latency, 3),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "finished_at": now_iso(),
        }

    completed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_job = {ex.submit(worker, job): job for job in tasks}
        for fut in cf.as_completed(future_to_job):
            job_row, job_model = future_to_job[fut]
            try:
                row_out = fut.result()
            except Exception as e:
                row_out = {
                    "run_id": run_id,
                    "runtime_id": job_row.get("runtime_id", ""),
                    "instance_id": job_row.get("instance_id", ""),
                    "benchmark_id": job_row.get("benchmark_id", ""),
                    "category": job_row.get("category", ""),
                    "subcategory": job_row.get("subcategory", ""),
                    "question_type": job_row.get("question_type", ""),
                    "model_name": job_model,
                    "model_snapshot": job_model,
                    "api_response_model": "",
                    "finish_reason": "",
                    "provider": MODEL_PROVIDER.get(job_model, ""),
                    "api_key_label": key_label_for_model("test", job_model),
                    "system_prompt_version": sha256_text(system_prompt)[:12],
                    "context": job_row.get("context", ""),
                    "prompt": job_row.get("prompt", ""),
                    "raw_response": "",
                    "parsed_answer": "",
                    "parse_success": False,
                    "error_message": repr(e),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "usage_estimated": False,
                    "latency_sec": 0,
                    "temperature": temperature,
                    "max_tokens": open_max_tokens if job_row.get("question_type") == "open_task" else closed_max_tokens,
                    "finished_at": now_iso(),
                }
            append_csv_row(out_path, row_out, RAW_OUTPUT_FIELDS, lock)
            if row_out.get("error_message"):
                append_csv_row(err_path, {
                    "run_id": run_id, "phase": "inference", "runtime_id": row_out.get("runtime_id"),
                    "instance_id": row_out.get("instance_id"), "model_name": row_out.get("model_name"),
                    "judge_model": "", "attempt": "final", "error_type": "api_error_or_worker_exception",
                    "error_message": row_out.get("error_message"), "finished_at": row_out.get("finished_at"),
                }, ERROR_FIELDS, err_lock)
            completed += 1
            elapsed = time.time() - stage_started_at
            avg_per_task = elapsed / completed if completed else 0.0
            eta = avg_per_task * (len(tasks) - completed)
            pct = completed / len(tasks) * 100 if tasks else 100.0
            has_error = bool(row_out.get("error_message"))
            should_log = (
                completed == 1
                or completed % progress_interval == 0
                or completed == len(tasks)
                or has_error
            )
            if should_log:
                err_short = f" | error={str(row_out.get('error_message'))[:120]}" if has_error else ""
                ok = not has_error and bool(str(row_out.get("raw_response", "")).strip())
                print(
                    f"[模型作答] 进度 {completed}/{len(tasks)} ({pct:.1f}%)"
                    f" | 最近完成 {row_out.get('runtime_id', '')} | {row_out.get('model_name', '')}"
                    f" | {row_out.get('question_type', '')} | ok={ok}"
                    f" | parse_success={row_out.get('parse_success', '')}"
                    f" | finish_reason={row_out.get('finish_reason', '') or 'NA'}"
                    f" | latency={row_out.get('latency_sec', '')}s"
                    f" | elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}{err_short}",
                    flush=True,
                )

# ----------------------------
# 第 2 阶段：封闭题 exact match 判分
# ----------------------------

def score_closed_items(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    raw_path = out_dir / "01_model_outputs_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError("01_model_outputs_raw.csv not found. Run inference first.")
    raw = dedupe_inference_raw(pd.read_csv(raw_path, encoding="utf-8-sig"))
    meta_cols = ["runtime_id", "instance_id", "benchmark_id", "category", "subcategory", "question_type", "prompt", "reference_answer_or_key"]
    meta = df[meta_cols].copy()
    closed = raw[raw["question_type"] == "closed_qa"].merge(meta, on=["runtime_id", "instance_id", "benchmark_id", "category", "subcategory", "question_type", "prompt"], how="left")

    rows = []
    for _, r in closed.iterrows():
        valid_options = option_letters_from_prompt(str(r.get("prompt", "")))
        ref = extract_answer_letters(r.get("reference_answer_or_key", ""), valid_options=valid_options, reference_mode=True)
        parsed = str(r.get("parsed_answer", "") or "")
        if not parsed:
            parsed = extract_answer_letters(r.get("raw_response", ""), valid_options=valid_options, reference_mode=False)
        correct, valid = exact_match_score(parsed, ref)
        rows.append({
            "runtime_id": r["runtime_id"],
            "instance_id": r["instance_id"],
            "benchmark_id": r["benchmark_id"],
            "category": r["category"],
            "subcategory": r.get("subcategory", ""),
            "model_name": r["model_name"],
            "question_type": "closed_qa",
            "standard_answer": ref,
            "parsed_answer": parsed,
            "is_correct": correct,
            "score_raw": correct,
            "score_norm": correct,
            "answer_valid": valid,
            "judge_rule": "exact_match_single_choice",
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "02_closed_item_scores.csv", index=False, encoding="utf-8-sig")
    return out

# ----------------------------
# 第 3 阶段：开放题三模型 LLM-jury 评分
# ----------------------------

JURY_FIELDS = [
    "run_id", "runtime_id", "instance_id", "benchmark_id", "category", "subcategory",
    "model_under_eval", "judge_model", "judge_snapshot", "judge_api_response_model", "judge_finish_reason", "api_key_label", "accuracy_score", "completeness_score",
    "clarity_score", "safety_score", "overall_score", "judge_reason", "judge_raw_response",
    "judge_parse_success", "judge_input_tokens", "judge_output_tokens", "judge_total_tokens",
    "usage_estimated", "judge_cost", "finished_at",
]


def run_jury_scoring(
    df: pd.DataFrame,
    clients: Dict[Tuple[str, str], APIYiClient],
    out_dir: Path,
    run_id: str,
    workers: int,
    force: bool,
    price_df: pd.DataFrame,
    jury_max_tokens: int,
) -> None:
    raw_path = out_dir / "01_model_outputs_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError("01_model_outputs_raw.csv not found. Run inference first.")
    raw = dedupe_inference_raw(pd.read_csv(raw_path, encoding="utf-8-sig"))
    open_raw = raw[(raw["question_type"] == "open_task") & (raw["raw_response"].fillna("") != "")].copy()
    meta = df.set_index("runtime_id")

    out_path = out_dir / "03_open_jury_scores_raw.csv"
    err_path = out_dir / "11_error_retry_log.csv"
    lock = threading.Lock()
    err_lock = threading.Lock()
    done = set() if force else load_successful_jury_key_set(out_path)

    tasks = []
    for _, rr in open_raw.iterrows():
        runtime_id = rr["runtime_id"]
        if runtime_id not in meta.index:
            continue
        row = meta.loc[runtime_id]
        # 把 runtime_id 恢复到 row 里，方便构建 jury prompt。
        row = row.copy()
        row["runtime_id"] = runtime_id
        for judge in JUDGE_MODELS:
            key = (str(runtime_id), str(rr["model_name"]), judge)
            if key not in done:
                tasks.append((row, rr, judge))

    total_possible = len(open_raw) * len(JUDGE_MODELS)
    skipped_successful = max(0, total_possible - len(tasks)) if not force else 0
    print(
        f"[开放题 jury] 阶段开始：待请求 {len(tasks)} 次；已成功跳过 {skipped_successful} 次；"
        f"理论总请求 {total_possible} 次；workers={workers}",
        flush=True,
    )
    if not tasks:
        return

    stage_started_at = time.time()
    progress_interval = 1 if len(tasks) <= 50 else 10

    def worker(job: Tuple[pd.Series, pd.Series, str]) -> Dict[str, Any]:
        row, rr, judge = job
        t0 = time.time()
        jury_user_prompt = build_jury_prompt(row, str(rr["raw_response"]))
        messages = [
            {"role": "system", "content": JURY_SYSTEM_PROMPT},
            {"role": "user", "content": jury_user_prompt},
        ]
        client = clients[("judge", judge)]
        res = client.chat_completion(model=judge, messages=messages, temperature=0.0, max_tokens=jury_max_tokens)
        latency = time.time() - t0
        content = res.get("content") or ""
        usage_estimated = False
        in_tok = res.get("prompt_tokens")
        out_tok = res.get("completion_tokens")
        tot_tok = res.get("total_tokens")
        if in_tok is None or out_tok is None:
            usage_estimated = True
            in_tok = approx_tokens(JURY_SYSTEM_PROMPT + "\n" + jury_user_prompt)
            out_tok = approx_tokens(content)
            tot_tok = in_tok + out_tok
        js, parse_ok = parse_jury_score_object(content)
        acc = clamp_score(js.get("accuracy_score"))
        comp = clamp_score(js.get("completeness_score"))
        clarity = clamp_score(js.get("clarity_score"))
        safety = clamp_score(js.get("safety_score"))
        overall = clamp_score(js.get("overall_score"))
        if overall is None:
            dims = [x for x in [acc, comp, clarity, safety] if x is not None]
            overall = sum(dims) / len(dims) if dims else None
        judge_cost = model_cost(judge, in_tok, out_tok, price_df)
        err = "" if res.get("ok") else (res.get("error") or "unknown error")
        return {
            "run_id": run_id,
            "runtime_id": row["runtime_id"],
            "instance_id": row["instance_id"],
            "benchmark_id": row["benchmark_id"],
            "category": row["category"],
            "subcategory": row.get("subcategory", ""),
            "model_under_eval": rr["model_name"],
            "judge_model": judge,
            "judge_snapshot": judge,
            "judge_api_response_model": res.get("response_model", ""),
            "judge_finish_reason": res.get("finish_reason", ""),
            "api_key_label": key_label_for_model("judge", judge),
            "accuracy_score": acc,
            "completeness_score": comp,
            "clarity_score": clarity,
            "safety_score": safety,
            "overall_score": overall,
            "judge_reason": js.get("rationale", "") if isinstance(js, dict) else "",
            "judge_raw_response": content,
            "judge_parse_success": parse_ok,
            "judge_input_tokens": in_tok,
            "judge_output_tokens": out_tok,
            "judge_total_tokens": tot_tok,
            "usage_estimated": usage_estimated,
            "judge_cost": judge_cost,
            "finished_at": now_iso(),
            "_error_message": err,
            "_latency_sec": round(latency, 3),
        }

    completed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_job = {ex.submit(worker, job): job for job in tasks}
        for fut in cf.as_completed(future_to_job):
            job_row, job_rr, job_judge = future_to_job[fut]
            try:
                row_out = fut.result()
            except Exception as e:
                row_out = {
                    "run_id": run_id,
                    "runtime_id": job_row.get("runtime_id", ""),
                    "instance_id": job_row.get("instance_id", ""),
                    "benchmark_id": job_row.get("benchmark_id", ""),
                    "category": job_row.get("category", ""),
                    "subcategory": job_row.get("subcategory", ""),
                    "model_under_eval": job_rr.get("model_name", ""),
                    "judge_model": job_judge,
                    "judge_snapshot": job_judge,
                    "judge_api_response_model": "",
                    "judge_finish_reason": "",
                    "api_key_label": key_label_for_model("judge", job_judge),
                    "accuracy_score": None,
                    "completeness_score": None,
                    "clarity_score": None,
                    "safety_score": None,
                    "overall_score": None,
                    "judge_reason": "",
                    "judge_raw_response": "",
                    "judge_parse_success": False,
                    "judge_input_tokens": 0,
                    "judge_output_tokens": 0,
                    "judge_total_tokens": 0,
                    "usage_estimated": False,
                    "judge_cost": 0,
                    "finished_at": now_iso(),
                    "_error_message": repr(e),
                }
            append_csv_row(out_path, row_out, JURY_FIELDS, lock)
            if row_out.get("_error_message"):
                append_csv_row(err_path, {
                    "run_id": run_id, "phase": "jury", "runtime_id": row_out.get("runtime_id"),
                    "instance_id": row_out.get("instance_id"), "model_name": row_out.get("model_under_eval"),
                    "judge_model": row_out.get("judge_model"), "attempt": "final", "error_type": "api_error_or_worker_exception",
                    "error_message": row_out.get("_error_message"), "finished_at": row_out.get("finished_at"),
                }, ERROR_FIELDS, err_lock)
            completed += 1
            elapsed = time.time() - stage_started_at
            avg_per_task = elapsed / completed if completed else 0.0
            eta = avg_per_task * (len(tasks) - completed)
            pct = completed / len(tasks) * 100 if tasks else 100.0
            has_error = bool(row_out.get("_error_message"))
            should_log = (
                completed == 1
                or completed % progress_interval == 0
                or completed == len(tasks)
                or has_error
            )
            if should_log:
                err_short = f" | error={str(row_out.get('_error_message'))[:120]}" if has_error else ""
                print(
                    f"[开放题 jury] 进度 {completed}/{len(tasks)} ({pct:.1f}%)"
                    f" | 最近完成 {row_out.get('runtime_id', '')} | 被评模型={row_out.get('model_under_eval', '')}"
                    f" | judge={row_out.get('judge_model', '')}"
                    f" | ok={not has_error} | parse_success={row_out.get('judge_parse_success', '')}"
                    f" | accuracy={row_out.get('accuracy_score', '')}"
                    f" | finish_reason={row_out.get('judge_finish_reason', '') or 'NA'}"
                    f" | latency={row_out.get('_latency_sec', '')}s"
                    f" | elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}{err_short}",
                    flush=True,
                )


def aggregate_open_scores(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "03_open_jury_scores_raw.csv"
    if not path.exists():
        raise FileNotFoundError("03_open_jury_scores_raw.csv not found. Run jury first.")
    j = dedupe_jury_raw(pd.read_csv(path, encoding="utf-8-sig"))
    score_cols = ["accuracy_score", "completeness_score", "clarity_score", "safety_score", "overall_score"]
    for c in score_cols:
        if c not in j.columns:
            j[c] = pd.NA
        j[c] = pd.to_numeric(j[c], errors="coerce")

    # judge_parse_success 从 CSV 读回后可能是 bool、True/False 字符串或 1/0；统一为布尔。
    j["judge_parse_success_bool"] = truthy_series(j.get("judge_parse_success", pd.Series(False, index=j.index)))
    core_score_ok = j[["accuracy_score", "completeness_score", "clarity_score", "overall_score"]].notna().any(axis=1)
    j["judge_valid"] = j["judge_parse_success_bool"] & core_score_ok

    group_cols = ["runtime_id", "instance_id", "benchmark_id", "category", "subcategory", "model_under_eval"]
    agg = j.groupby(group_cols, dropna=False).agg(
        n_judge_rows=("judge_model", "size"),
        n_judges=("judge_model", "nunique"),
        n_valid_judges=("judge_valid", "sum"),
        mean_accuracy=("accuracy_score", "mean"),
        mean_completeness=("completeness_score", "mean"),
        mean_clarity=("clarity_score", "mean"),
        mean_safety=("safety_score", "mean"),
        mean_overall_raw=("overall_score", "mean"),
        jury_sd=("overall_score", "std"),
        jury_min=("overall_score", "min"),
        jury_max=("overall_score", "max"),
        jury_parse_success_rate=("judge_parse_success_bool", "mean"),
    ).reset_index()
    agg = agg.rename(columns={"model_under_eval": "model_name"})
    agg["question_type"] = "open_task"
    agg["all_three_judges_valid"] = agg["n_valid_judges"].fillna(0).astype(int) >= len(JUDGE_MODELS)
    # 为了更贴近 MedHELM，开放题主评分使用三个核心维度的平均值：
    # accuracy、completeness、clarity。safety_score 保留为补充安全性维度，不直接进入主 leaderboard。
    agg["mean_primary_raw"] = agg[["mean_accuracy", "mean_completeness", "mean_clarity"]].mean(axis=1)
    agg["mean_primary_raw"] = agg["mean_primary_raw"].fillna(agg["mean_overall_raw"])
    # 没有任何有效 judge 的项目不应进入主分析，保留为 NaN，QC 表会提示缺失。
    agg.loc[agg["n_valid_judges"].fillna(0).astype(int) == 0, "mean_primary_raw"] = pd.NA
    # 将 1-5 分归一化到 0-1，方便和封闭题 accuracy 放在同一尺度。
    agg["score_norm"] = (agg["mean_primary_raw"] - 1.0) / 4.0
    agg["score_norm"] = agg["score_norm"].clip(lower=0, upper=1)
    agg.to_csv(out_dir / "04_open_item_scores.csv", index=False, encoding="utf-8-sig")
    return agg

# ----------------------------
# 第 4 阶段：聚合生成论文需要的结果表
# ----------------------------

def compute_win_stats(bench_wide: pd.DataFrame, benchmark_cols: List[str]) -> pd.DataFrame:
    """计算参考文献风格的 pairwise win rate。

    主列采用严格胜出口径：normalized score > rival 时计为 win。
    同时保留两种敏感性列：tie=0.5 和 tie=win，便于 Methods 或 Supplementary Table 报告。
    """
    models = bench_wide["Model"].tolist()
    vals = bench_wide.set_index("Model")[benchmark_cols].apply(pd.to_numeric, errors="coerce")
    mean_rates_strict, win_sds_strict = [], []
    mean_rates_half, win_sds_half = [], []
    mean_rates_tie_win, win_sds_tie_win = [], []
    for m in models:
        per_strict, per_half, per_tie_win = [], [], []
        total_strict = total_half = total_tie_win = 0.0
        total_comparisons = 0
        for b in benchmark_cols:
            v = vals.loc[m, b]
            if pd.isna(v):
                continue
            wins_strict = wins_half = wins_tie_win = 0.0
            comps = 0
            for other in models:
                if other == m:
                    continue
                ov = vals.loc[other, b]
                if pd.isna(ov):
                    continue
                point_strict = 1.0 if v > ov else 0.0
                point_half = 1.0 if v > ov else (0.5 if v == ov else 0.0)
                point_tie_win = 1.0 if v >= ov else 0.0
                total_strict += point_strict
                total_half += point_half
                total_tie_win += point_tie_win
                total_comparisons += 1
                wins_strict += point_strict
                wins_half += point_half
                wins_tie_win += point_tie_win
                comps += 1
            if comps:
                per_strict.append(wins_strict / comps)
                per_half.append(wins_half / comps)
                per_tie_win.append(wins_tie_win / comps)
        mean_rates_strict.append(total_strict / total_comparisons if total_comparisons else float("nan"))
        win_sds_strict.append(float(pd.Series(per_strict).std(ddof=0)) if per_strict else float("nan"))
        mean_rates_half.append(total_half / total_comparisons if total_comparisons else float("nan"))
        win_sds_half.append(float(pd.Series(per_half).std(ddof=0)) if per_half else float("nan"))
        mean_rates_tie_win.append(total_tie_win / total_comparisons if total_comparisons else float("nan"))
        win_sds_tie_win.append(float(pd.Series(per_tie_win).std(ddof=0)) if per_tie_win else float("nan"))
    return pd.DataFrame({
        "Model": models,
        "Mean win rate": mean_rates_strict,
        "Win SD": win_sds_strict,
        "Mean win rate (tie=0.5)": mean_rates_half,
        "Win SD (tie=0.5)": win_sds_half,
        "Mean win rate (tie=win)": mean_rates_tie_win,
        "Win SD (tie=win)": win_sds_tie_win,
    })


def build_pairwise_win_tables(bench_wide: pd.DataFrame, benchmark_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """输出模型两两比较表。

    16_pairwise_win_matrix.csv 使用主口径：score > opponent。
    17_pairwise_win_long.csv 同时保存 tie=0.5 和 tie=win 的敏感性列。
    """
    models = bench_wide["Model"].tolist()
    vals = bench_wide.set_index("Model")[benchmark_cols].apply(pd.to_numeric, errors="coerce")
    matrix_rows, long_rows = [], []
    for m in models:
        row = {"Model": m}
        for other in models:
            if other == m:
                row[other] = None
                continue
            pts_strict, pts_half, pts_tie_win, deltas = [], [], [], []
            for b in benchmark_cols:
                v, ov = vals.loc[m, b], vals.loc[other, b]
                if pd.isna(v) or pd.isna(ov):
                    continue
                pts_strict.append(1.0 if v > ov else 0.0)
                pts_half.append(1.0 if v > ov else (0.5 if v == ov else 0.0))
                pts_tie_win.append(1.0 if v >= ov else 0.0)
                deltas.append(v - ov)
            win_strict = float(pd.Series(pts_strict).mean()) if pts_strict else float("nan")
            win_half = float(pd.Series(pts_half).mean()) if pts_half else float("nan")
            win_tie_win = float(pd.Series(pts_tie_win).mean()) if pts_tie_win else float("nan")
            row[other] = win_strict
            long_rows.append({
                "model": m,
                "opponent": other,
                "n_benchmarks_compared": len(pts_strict),
                "win_rate_strict_gt": win_strict,
                "win_rate_tie_half": win_half,
                "win_rate_tie_win_ge": win_tie_win,
                # 保留旧列名作为兼容列，方便已经写好的临时分析脚本不立刻失效。
                "win_rate_reference_ge_tie_win": win_tie_win,
                "mean_score_delta": float(pd.Series(deltas).mean()) if deltas else float("nan"),
            })
        matrix_rows.append(row)
    return pd.DataFrame(matrix_rows), pd.DataFrame(long_rows)


def write_preflight_tables(df: pd.DataFrame, out_dir: Path, price_df: pd.DataFrame, system_prompt: str, args: argparse.Namespace, evaluation_config_hash: str) -> None:
    expected_inference = len(df) * len(TEST_MODELS)
    expected_jury = int((df["question_type"] == "open_task").sum()) * len(TEST_MODELS) * len(JUDGE_MODELS)
    checks = []
    def add_check(name: str, status: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if status else "FAIL", "detail": detail})
    add_check("input_file_exists", Path(args.input_csv).exists(), str(args.input_csv))
    add_check("mapping_file_exists", Path(args.mapping_xlsx).exists(), str(args.mapping_xlsx))
    add_check("benchmark_file_exists", Path(args.benchmark_xlsx).exists(), str(args.benchmark_xlsx))
    add_check("prompt_file_exists", Path(args.prompt_txt).exists(), str(args.prompt_txt))
    add_check("has_240_items", len(df) == 240, f"observed_items={len(df)}")
    add_check("runtime_id_unique", not df["runtime_id"].duplicated().any(), f"duplicate_runtime_id={int(df['runtime_id'].duplicated().sum())}")
    add_check("instance_id_mapped", not df["instance_id"].isna().any(), f"missing_instance_id={int(df['instance_id'].isna().sum())}")
    add_check("question_types_valid", set(df["question_type"].dropna().unique()).issubset({"closed_qa", "open_task"}), f"types={sorted(df['question_type'].dropna().unique().tolist())}")
    q_counts = df["question_type"].value_counts(dropna=False).to_dict()
    add_check("question_type_counts_expected", q_counts.get("closed_qa", 0) == 138 and q_counts.get("open_task", 0) == 102, f"counts={q_counts}")
    add_check("runtime_id_format", df["runtime_id"].astype(str).str.fullmatch(r"CASE-\d{4}").all(), "expected format CASE-0001 ... CASE-0240")
    add_check("runtime_id_range", set(df["runtime_id"].astype(str)) == {f"CASE-{i:04d}" for i in range(1, len(df) + 1)}, f"expected contiguous CASE-0001..CASE-{len(df):04d}")
    core_meta_cols = [c for c in ["benchmark_id", "category", "subcategory", "reference_answer_or_key", "prompt", "context"] if c in df.columns]
    missing_core = int(df[core_meta_cols].isna().any(axis=1).sum()) if core_meta_cols else len(df)
    add_check("core_metadata_complete", missing_core == 0, f"rows_with_missing_core_metadata={missing_core}")
    closed_ref_bad = []
    for _, rr in df[df["question_type"] == "closed_qa"].iterrows():
        ref = extract_answer_letters(rr.get("reference_answer_or_key", ""), valid_options=option_letters_from_prompt(str(rr.get("prompt", ""))), reference_mode=True)
        if not ref:
            closed_ref_bad.append(str(rr.get("runtime_id", "")))
    add_check("closed_reference_answers_parseable", len(closed_ref_bad) == 0, f"unparseable_closed_refs={len(closed_ref_bad)}; examples={closed_ref_bad[:5]}")
    add_check("has_12_benchmarks", df["benchmark_id"].nunique() == 12, f"observed_benchmarks={df['benchmark_id'].nunique()}")
    add_check("has_9_test_models", len(TEST_MODELS) == 9, f"test_models={len(TEST_MODELS)}")
    add_check("has_3_judge_models", len(JUDGE_MODELS) == 3, f"judge_models={len(JUDGE_MODELS)}")
    add_check("judge_models_subset_of_test_models", set(JUDGE_MODELS).issubset(set(TEST_MODELS)), f"judges={JUDGE_MODELS}")
    add_check("system_prompt_nonempty", len(system_prompt.strip()) > 0, f"chars={len(system_prompt)}; sha256={sha256_text(system_prompt)[:12]}")
    add_check("price_rows_cover_models", set(TEST_MODELS + JUDGE_MODELS).issubset(set(price_df["model_name"].astype(str))), f"price_rows={price_df.shape[0]}")
    price_needed = price_df[price_df["model_name"].astype(str).isin(set(TEST_MODELS + JUDGE_MODELS))].copy()
    price_needed["input_price_num"] = pd.to_numeric(price_needed.get("input_price_per_1m"), errors="coerce")
    price_needed["output_price_num"] = pd.to_numeric(price_needed.get("output_price_per_1m"), errors="coerce")
    missing_price_models = price_needed.loc[
        price_needed["input_price_num"].isna() | price_needed["output_price_num"].isna(),
        "model_name"
    ].astype(str).tolist()
    add_check("price_values_complete", len(missing_price_models) == 0, f"missing_or_non_numeric_prices={missing_price_models}")
    add_check("expected_api_calls", True, f"inference_calls={expected_inference}; jury_calls={expected_jury}; total={expected_inference + expected_jury}")
    add_check("evaluation_config_hash_written", bool(evaluation_config_hash), f"evaluation_config_hash={evaluation_config_hash[:12]}")
    pd.DataFrame(checks).to_csv(out_dir / "00_preflight_validation.csv", index=False, encoding="utf-8-sig")
    manifest = df.groupby(["benchmark_id", "benchmark_display_name", "category"], dropna=False).agg(
        subcategories=("subcategory", lambda s: " | ".join(sorted({str(x) for x in s.dropna() if str(x).strip()}))),
        total_instances=("runtime_id", "nunique"),
        closed_instances=("question_type", lambda s: int((s == "closed_qa").sum())),
        open_instances=("question_type", lambda s: int((s == "open_task").sum())),
        basic_instances=("complexity", lambda s: int((s == "basic").sum())),
        intermediate_instances=("complexity", lambda s: int((s == "intermediate").sum())),
        advanced_instances=("complexity", lambda s: int((s == "advanced").sum())),
    ).reset_index()
    manifest.to_csv(out_dir / "00_benchmark_manifest.csv", index=False, encoding="utf-8-sig")

    # Supplementary Table 2 源数据：240 个实例按 category/subcategory/benchmark/question_type/complexity 的分布。
    dist_cols = [c for c in ["category", "subcategory", "benchmark_id", "benchmark_display_name", "question_type", "complexity"] if c in df.columns]
    if dist_cols:
        dist = df.groupby(dist_cols, dropna=False).agg(n_items=("runtime_id", "nunique")).reset_index()
        dist.to_csv(out_dir / "24_instance_distribution_summary.csv", index=False, encoding="utf-8-sig")

    write_jury_prompt_and_rubric(out_dir, system_prompt)


def write_jury_prompt_and_rubric(out_dir: Path, system_prompt: str) -> None:
    """导出 Supplementary Fig.1 / Methods 可直接引用的 LLM-jury prompt 与评分规则。"""
    text = f"""TBI-NeuroHELM LLM-jury prompt and rubric
Generated by: {SCRIPT_VERSION}
Generated at: {now_iso()}

1. System prompt for tested models
----------------------------------
{system_prompt.strip()}

2. System prompt for LLM-jury models
-------------------------------------
{JURY_SYSTEM_PROMPT.strip()}

3. Open-task scoring anchors
----------------------------
{OPEN_TASK_SCORING_ANCHORS.strip()}

4. User prompt template for each LLM-jury request
-------------------------------------------------
The script builds one scoring prompt per open-task answer using the following fields:
- Case ID
- Benchmark
- Category
- Subcategory
- Context
- Task prompt
- Reference answer or scoring rubric
- Candidate response

The jury model is instructed to output JSON only, with these exact fields:
{{
  "accuracy_score": integer 1-5,
  "completeness_score": integer 1-5,
  "clarity_score": integer 1-5,
  "safety_score": integer 1-5,
  "overall_score": integer 1-5,
  "rationale": brief string
}}

The parser also accepts common aliases such as accuracy/completeness/clarity/safety/overall, but the formal schema used in Methods and Supplementary material is the *_score schema above.

5. Primary open-task score definition
-------------------------------------
For each open-task answer, the primary raw score is the mean of Accuracy, Completeness, and Clarity across valid jury ratings.
Safety and Overall are retained as auxiliary dimensions and do not enter the primary leaderboard.
The normalized open-task score is calculated as: score_norm = (mean_primary_raw - 1) / 4.
"""
    (out_dir / "23_jury_prompt_and_rubric.txt").write_text(text, encoding="utf-8")


def compute_subset_leaderboard_rows(closed_items: pd.DataFrame, jury: pd.DataFrame, benchmark_order: List[str]) -> pd.DataFrame:
    if jury.empty:
        return pd.DataFrame()
    rows = []
    for c in ["accuracy_score", "completeness_score", "clarity_score", "safety_score", "overall_score"]:
        if c in jury.columns:
            jury[c] = pd.to_numeric(jury[c], errors="coerce")
    jury["judge_parse_success_bool"] = truthy_series(jury.get("judge_parse_success", pd.Series(False, index=jury.index)))
    jury["judge_valid"] = jury["judge_parse_success_bool"] & jury[["accuracy_score", "completeness_score", "clarity_score", "overall_score"]].notna().any(axis=1)
    jury = jury[jury["judge_valid"]].copy()
    if jury.empty:
        return pd.DataFrame()
    judges_available = [j for j in JUDGE_MODELS if j in set(jury["judge_model"].astype(str))]
    for r in range(1, len(judges_available) + 1):
        for subset in itertools.combinations(judges_available, r):
            jj = jury[jury["judge_model"].isin(subset)].copy()
            if jj.empty:
                continue
            group_cols = ["runtime_id", "instance_id", "benchmark_id", "category", "subcategory", "model_under_eval"]
            oo = jj.groupby(group_cols, dropna=False).agg(
                n_judges=("judge_model", "nunique"),
                mean_accuracy=("accuracy_score", "mean"),
                mean_completeness=("completeness_score", "mean"),
                mean_clarity=("clarity_score", "mean"),
                mean_safety=("safety_score", "mean"),
                mean_overall_raw=("overall_score", "mean"),
            ).reset_index().rename(columns={"model_under_eval": "model_name"})
            oo["mean_primary_raw"] = oo[["mean_accuracy", "mean_completeness", "mean_clarity"]].mean(axis=1).fillna(oo["mean_overall_raw"])
            oo["score_norm"] = ((oo["mean_primary_raw"] - 1.0) / 4.0).clip(lower=0, upper=1)
            oo["question_type"] = "open_task"
            oo2 = oo[["runtime_id", "instance_id", "benchmark_id", "category", "subcategory", "model_name", "question_type", "score_norm"]].copy()
            combo_items = pd.concat([closed_items, oo2], ignore_index=True) if not closed_items.empty else oo2
            bench2 = combo_items.groupby(["model_name", "benchmark_id"], dropna=False).agg(benchmark_score=("score_norm", "mean")).reset_index()
            bench2["benchmark_col"] = bench2["benchmark_id"].astype(str) + " - Score"
            wide2 = bench2.pivot_table(index="model_name", columns="benchmark_col", values="benchmark_score", aggfunc="mean").reset_index().rename(columns={"model_name": "Model"})
            bcols = [f"{b} - Score" for b in benchmark_order if f"{b} - Score" in wide2.columns]
            if not bcols:
                continue
            wide2["Macro-average"] = wide2[bcols].mean(axis=1)
            wide2["SD"] = wide2[bcols].std(axis=1, ddof=0)
            win2 = compute_win_stats(wide2, bcols)
            wide2 = wide2.merge(win2, on="Model", how="left").sort_values(["Mean win rate", "Macro-average"], ascending=False).reset_index(drop=True)
            wide2["rank"] = range(1, len(wide2) + 1)
            for _, rr in wide2.iterrows():
                rows.append({"jury_subset": "+".join(subset), "subset_size": r, "model_name": rr["Model"], "rank": int(rr["rank"]), "mean_win_rate": rr["Mean win rate"], "win_sd": rr["Win SD"], "macro_average": rr["Macro-average"], "macro_sd": rr["SD"], "mean_win_rate_tie_half": rr.get("Mean win rate (tie=0.5)"), "mean_win_rate_tie_win": rr.get("Mean win rate (tie=win)")})
    return pd.DataFrame(rows)


def summarize_jury_robustness(combo_rows: pd.DataFrame) -> pd.DataFrame:
    """把 20_jury_combination_robustness.csv 压缩成投稿友好的 summary 表。"""
    if combo_rows.empty:
        return pd.DataFrame()
    full_subset = "+".join(JUDGE_MODELS)
    if full_subset not in set(combo_rows["jury_subset"].astype(str)):
        # 如果某些 judge 缺失，则用 subset_size 最大的组合作为参考。
        max_size = combo_rows["subset_size"].max()
        candidates = sorted(combo_rows.loc[combo_rows["subset_size"] == max_size, "jury_subset"].astype(str).unique().tolist())
        full_subset = candidates[0] if candidates else ""
    if not full_subset:
        return pd.DataFrame()
    ref = combo_rows[combo_rows["jury_subset"].astype(str) == full_subset].copy()
    ref_rank = ref.set_index("model_name")["rank"]
    ref_macro = ref.set_index("model_name")["macro_average"]
    ref_top1 = ref.sort_values(["rank", "macro_average"], ascending=[True, False])["model_name"].iloc[0] if not ref.empty else ""
    ref_top3 = set(ref.sort_values(["rank", "macro_average"], ascending=[True, False])["model_name"].head(3).tolist())
    rows = []
    for subset, sub in combo_rows.groupby("jury_subset", dropna=False):
        sub = sub.copy()
        common = [m for m in sub["model_name"].astype(str).tolist() if m in ref_rank.index]
        sub_rank = sub.set_index("model_name")["rank"]
        sub_macro = sub.set_index("model_name")["macro_average"]
        if len(common) >= 2:
            spearman_rank = sub_rank.loc[common].corr(ref_rank.loc[common], method="spearman")
            pearson_macro = sub_macro.loc[common].corr(ref_macro.loc[common], method="pearson")
        else:
            spearman_rank = float("nan")
            pearson_macro = float("nan")
        sub_sorted = sub.sort_values(["rank", "macro_average"], ascending=[True, False])
        top1 = sub_sorted["model_name"].iloc[0] if not sub_sorted.empty else ""
        top3 = set(sub_sorted["model_name"].head(3).tolist())
        rows.append({
            "jury_subset": subset,
            "subset_size": int(sub["subset_size"].iloc[0]) if "subset_size" in sub else None,
            "reference_subset": full_subset,
            "n_models": int(sub["model_name"].nunique()),
            "spearman_rank_corr_with_reference": spearman_rank,
            "pearson_macro_corr_with_reference": pearson_macro,
            "top1_model": top1,
            "top1_matches_reference": bool(top1 == ref_top1),
            "top3_overlap_with_reference": len(top3 & ref_top3) / 3 if ref_top3 else float("nan"),
            "mean_macro_average": pd.to_numeric(sub["macro_average"], errors="coerce").mean(),
            "mean_win_rate": pd.to_numeric(sub["mean_win_rate"], errors="coerce").mean(),
        })
    return pd.DataFrame(rows)



def aggregate_all_tables(df: pd.DataFrame, out_dir: Path, price_df: pd.DataFrame, run_id: str, system_prompt: str, args: argparse.Namespace) -> None:
    closed_path = out_dir / "02_closed_item_scores.csv"
    open_path = out_dir / "04_open_item_scores.csv"
    if not closed_path.exists():
        score_closed_items(df, out_dir)
    if not open_path.exists() and (out_dir / "03_open_jury_scores_raw.csv").exists():
        aggregate_open_scores(out_dir)

    item_frames = []
    if closed_path.exists():
        c = pd.read_csv(closed_path, encoding="utf-8-sig")
        c2 = c[["runtime_id", "instance_id", "benchmark_id", "category", "subcategory", "model_name", "question_type", "score_raw", "score_norm"]].copy()
        c2["metric"] = "Exact Match"
        c2["is_valid"] = c.get("answer_valid", True)
        item_frames.append(c2)
    if open_path.exists():
        o = pd.read_csv(open_path, encoding="utf-8-sig")
        if "mean_primary_raw" not in o.columns:
            o["mean_primary_raw"] = o[["mean_accuracy", "mean_completeness", "mean_clarity"]].mean(axis=1)
            o["mean_primary_raw"] = o["mean_primary_raw"].fillna(o.get("mean_overall_raw"))
        o2 = o[["runtime_id", "instance_id", "benchmark_id", "category", "subcategory", "model_name", "question_type", "mean_primary_raw", "score_norm"]].copy()
        o2 = o2.rename(columns={"mean_primary_raw": "score_raw"})
        o2["metric"] = "Jury Score"
        valid_col = "n_valid_judges" if "n_valid_judges" in o.columns else "n_judges"
        o2["is_valid"] = o[valid_col].fillna(0).astype(float) > 0
        item_frames.append(o2)
    if not item_frames:
        raise RuntimeError("No item score tables found.")

    item_all = pd.concat(item_frames, ignore_index=True)
    item_all.to_csv(out_dir / "05_item_scores_all.csv", index=False, encoding="utf-8-sig")

    # benchmark 分数必须按 model × benchmark 聚合。不能把 subcategory 放进 groupby，
    # 否则一个 benchmark 覆盖多个 subcategory 时会被拆成多行，Fig.3/Table 1 的宏平均和 win rate 会被二次平均污染。
    bench = item_all.groupby(["model_name", "benchmark_id", "category"], dropna=False).agg(
        n_items=("runtime_id", "nunique"),
        benchmark_score=("score_norm", "mean"),
        score_sd=("score_norm", "std"),
    ).reset_index()

    # 增加每个 benchmark 内封闭题/开放题数量和分数。
    type_agg = item_all.groupby(["model_name", "benchmark_id", "question_type"], dropna=False).agg(
        n=("runtime_id", "nunique"), score=("score_norm", "mean")
    ).reset_index()
    pivot_n = type_agg.pivot_table(index=["model_name", "benchmark_id"], columns="question_type", values="n", aggfunc="sum").reset_index()
    pivot_s = type_agg.pivot_table(index=["model_name", "benchmark_id"], columns="question_type", values="score", aggfunc="mean").reset_index()
    pivot_n = pivot_n.rename(columns={"closed_qa": "n_closed", "open_task": "n_open"})
    pivot_s = pivot_s.rename(columns={"closed_qa": "closed_score", "open_task": "open_score"})
    bench = bench.merge(pivot_n, on=["model_name", "benchmark_id"], how="left")
    bench = bench.merge(pivot_s, on=["model_name", "benchmark_id"], how="left")
    for col in ["n_closed", "n_open"]:
        if col not in bench.columns:
            bench[col] = 0
        bench[col] = bench[col].fillna(0).astype(int)

    # 补充显示名和该 benchmark 覆盖的 subcategory 列表。
    display_map = df.drop_duplicates("benchmark_id").set_index("benchmark_id")["benchmark_display_name"].to_dict()
    subcat_map = df.groupby("benchmark_id")["subcategory"].apply(
        lambda s: " | ".join(sorted({str(x) for x in s.dropna() if str(x).strip()}))
    ).to_dict()
    bench["benchmark_display_name"] = bench["benchmark_id"].map(display_map).fillna(bench["benchmark_id"])
    bench["subcategories"] = bench["benchmark_id"].map(subcat_map).fillna("")
    bench["benchmark_score_norm"] = bench["benchmark_score"]
    bench = bench[[
        "model_name", "benchmark_id", "benchmark_display_name", "category", "subcategories",
        "n_items", "n_closed", "n_open", "closed_score", "open_score", "benchmark_score",
        "benchmark_score_norm", "score_sd",
    ]]
    bench.to_csv(out_dir / "06_benchmark_scores_long.csv", index=False, encoding="utf-8-sig")

    # 生成宽表 leaderboard：每个模型一行，每个 benchmark 一列。
    bench["benchmark_col"] = bench["benchmark_id"].astype(str) + " - Score"
    wide = bench.pivot_table(index="model_name", columns="benchmark_col", values="benchmark_score_norm", aggfunc="mean").reset_index()
    wide = wide.rename(columns={"model_name": "Model"})
    # 按题库中的 benchmark 顺序排列列，而不是简单按字母排序；这样 Fig.3 更贴近 CDS→DOC→COM→RES 的文章逻辑。
    benchmark_order = df.drop_duplicates("benchmark_id")["benchmark_id"].astype(str).tolist()
    benchmark_cols = [f"{b} - Score" for b in benchmark_order if f"{b} - Score" in wide.columns]
    extra_benchmark_cols = [c for c in wide.columns if c not in ["Model"] + benchmark_cols]
    benchmark_cols = benchmark_cols + extra_benchmark_cols
    wide["Macro-average"] = wide[benchmark_cols].mean(axis=1)
    wide["SD"] = wide[benchmark_cols].std(axis=1, ddof=0)
    win = compute_win_stats(wide, benchmark_cols)
    wide = wide.merge(win, on="Model", how="left")
    # 增加模型厂商和模型快照。
    wide.insert(1, "Provider", wide["Model"].map(MODEL_PROVIDER).fillna(""))
    wide.insert(2, "Model snapshot", wide["Model"])
    # 按论文 Table 1 风格排序列：模型、厂商、快照、胜率、平均分、各 benchmark。
    wide = wide[["Model", "Provider", "Model snapshot", "Mean win rate", "Win SD", "Mean win rate (tie=0.5)", "Win SD (tie=0.5)", "Mean win rate (tie=win)", "Win SD (tie=win)", "Macro-average", "SD"] + benchmark_cols]
    wide = wide.sort_values(["Mean win rate", "Macro-average"], ascending=False)
    wide.to_csv(out_dir / "07_leaderboard.csv", index=False, encoding="utf-8-sig")

    # 成本表：作答成本 + 开放题 jury 成本。
    raw = dedupe_inference_raw(pd.read_csv(out_dir / "01_model_outputs_raw.csv", encoding="utf-8-sig")) if (out_dir / "01_model_outputs_raw.csv").exists() else pd.DataFrame()
    if not raw.empty:
        for c in ["input_tokens", "output_tokens", "total_tokens"]:
            if c in raw.columns:
                raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)
        raw["benchmark_cost"] = raw.apply(lambda r: model_cost(r["model_name"], r.get("input_tokens"), r.get("output_tokens"), price_df), axis=1)
        raw_cost = raw.groupby(["model_name", "provider", "benchmark_id"], dropna=False).agg(
            benchmark_input_tokens=("input_tokens", "sum"),
            benchmark_output_tokens=("output_tokens", "sum"),
            benchmark_cost=("benchmark_cost", "sum"),
        ).reset_index().rename(columns={"model_name": "Model", "provider": "Provider", "benchmark_id": "Benchmark"})
    else:
        raw_cost = pd.DataFrame(columns=["Model", "Provider", "Benchmark", "benchmark_input_tokens", "benchmark_output_tokens", "benchmark_cost"])

    jury_path = out_dir / "03_open_jury_scores_raw.csv"
    if jury_path.exists() and jury_path.stat().st_size > 0:
        jury = dedupe_jury_raw(pd.read_csv(jury_path, encoding="utf-8-sig"))
        # 这里不要直接沿用 03_open_jury_scores_raw.csv 里旧的 judge_cost，
        # 因为你可能是在跑完后才填写价格表，然后用 MODE="aggregate" 重新汇总。
        # 因此每次汇总都根据最新 price_df 和 token usage 重新计算 jury_cost。
        for c in ["judge_input_tokens", "judge_output_tokens", "judge_total_tokens"]:
            if c in jury.columns:
                jury[c] = pd.to_numeric(jury[c], errors="coerce").fillna(0)
        jury["judge_cost_recomputed"] = jury.apply(
            lambda r: model_cost(r["judge_model"], r.get("judge_input_tokens"), r.get("judge_output_tokens"), price_df),
            axis=1,
        )
        jury_cost = jury.groupby(["model_under_eval", "benchmark_id"], dropna=False).agg(
            jury_input_tokens=("judge_input_tokens", "sum"),
            jury_output_tokens=("judge_output_tokens", "sum"),
            jury_cost=("judge_cost_recomputed", "sum"),
        ).reset_index().rename(columns={"model_under_eval": "Model", "benchmark_id": "Benchmark"})
    else:
        jury = pd.DataFrame()
        jury_cost = pd.DataFrame(columns=["Model", "Benchmark", "jury_input_tokens", "jury_output_tokens", "jury_cost"])

    costs = raw_cost.merge(jury_cost, on=["Model", "Benchmark"], how="left")
    for c in ["jury_input_tokens", "jury_output_tokens", "jury_cost"]:
        if c not in costs.columns:
            costs[c] = 0
        costs[c] = costs[c].fillna(0)
    costs["Cost Input Output Tokens"] = costs["benchmark_cost"].fillna(0) + costs["jury_cost"].fillna(0)
    costs["total_input_tokens"] = costs["benchmark_input_tokens"].fillna(0) + costs["jury_input_tokens"].fillna(0)
    costs["total_output_tokens"] = costs["benchmark_output_tokens"].fillna(0) + costs["jury_output_tokens"].fillna(0)
    costs.to_csv(out_dir / "08_costs.csv", index=False, encoding="utf-8-sig")

    # Fig.5 和正文结果段落更方便使用的 model-level 成本-性能汇总表。
    cost_model = costs.groupby(["Model", "Provider"], dropna=False).agg(
        benchmark_input_tokens=("benchmark_input_tokens", "sum"),
        benchmark_output_tokens=("benchmark_output_tokens", "sum"),
        benchmark_cost_usd=("benchmark_cost", "sum"),
        jury_input_tokens=("jury_input_tokens", "sum"),
        jury_output_tokens=("jury_output_tokens", "sum"),
        jury_cost_usd=("jury_cost", "sum"),
        total_input_tokens=("total_input_tokens", "sum"),
        total_output_tokens=("total_output_tokens", "sum"),
        total_cost_usd=("Cost Input Output Tokens", "sum"),
    ).reset_index() if not costs.empty else pd.DataFrame(columns=["Model", "Provider"])
    perf_cols = [c for c in ["Model", "Provider", "Mean win rate", "Win SD", "Mean win rate (tie=0.5)", "Mean win rate (tie=win)", "Macro-average", "SD"] if c in wide.columns]
    model_summary = wide[perf_cols].copy().merge(cost_model, on=["Model", "Provider"], how="left")
    for c in ["benchmark_input_tokens", "benchmark_output_tokens", "benchmark_cost_usd", "jury_input_tokens", "jury_output_tokens", "jury_cost_usd", "total_input_tokens", "total_output_tokens", "total_cost_usd"]:
        if c not in model_summary.columns:
            model_summary[c] = 0
        model_summary[c] = pd.to_numeric(model_summary[c], errors="coerce").fillna(0)
    model_summary["n_cases_evaluated"] = len(df)
    model_summary["cost_per_240_cases_usd"] = model_summary["total_cost_usd"]
    model_summary["cost_per_1000_cases_usd"] = model_summary["total_cost_usd"] / max(len(df), 1) * 1000
    model_summary["cost_per_macro_average_point_usd"] = model_summary.apply(
        lambda r: r["total_cost_usd"] / r["Macro-average"] if pd.notna(r.get("Macro-average")) and r.get("Macro-average") not in [0, "0"] else float("nan"),
        axis=1,
    ) if "Macro-average" in model_summary.columns else float("nan")
    model_summary.to_csv(out_dir / "22_model_cost_performance_summary.csv", index=False, encoding="utf-8-sig")

    # 额外输出按 API key 用途标签汇总的成本表，方便你和 API易后台 12 个 key 的账单逐一核对。
    key_cost_frames = []
    if not raw.empty:
        key_raw = raw.copy()
        key_raw["phase"] = "test"
        key_raw["cost_usd"] = key_raw["benchmark_cost"]
        key_raw = key_raw.groupby(["phase", "api_key_label", "model_name"], dropna=False).agg(
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            cost_usd=("cost_usd", "sum"),
            n_requests=("runtime_id", "count"),
        ).reset_index()
        key_cost_frames.append(key_raw)
    if jury_path.exists() and jury_path.stat().st_size > 0 and not jury.empty:
        key_jury = jury.copy()
        key_jury["phase"] = "judge"
        key_jury["model_name"] = key_jury["judge_model"]
        key_jury["cost_usd"] = key_jury["judge_cost_recomputed"]
        key_jury = key_jury.groupby(["phase", "api_key_label", "model_name"], dropna=False).agg(
            input_tokens=("judge_input_tokens", "sum"),
            output_tokens=("judge_output_tokens", "sum"),
            total_tokens=("judge_total_tokens", "sum"),
            cost_usd=("cost_usd", "sum"),
            n_requests=("runtime_id", "count"),
        ).reset_index()
        key_cost_frames.append(key_jury)
    if key_cost_frames:
        pd.concat(key_cost_frames, ignore_index=True).to_csv(out_dir / "08b_costs_by_api_key_label.csv", index=False, encoding="utf-8-sig")

    # 额外的论文图表/正文源数据表。
    # category 主分数按 benchmark 等权平均，更接近参考文献宏平均口径；同时保留 item-weighted 分数供敏感性分析。
    category_macro = bench.groupby(["model_name", "category"], dropna=False).agg(
        n_benchmarks=("benchmark_id", "nunique"),
        category_score=("benchmark_score_norm", "mean"),
        category_sd=("benchmark_score_norm", "std"),
    ).reset_index()
    category_item = item_all.groupby(["model_name", "category"], dropna=False).agg(
        n_items=("runtime_id", "nunique"),
        category_score_item_weighted=("score_norm", "mean"),
    ).reset_index()
    category_scores = category_macro.merge(category_item, on=["model_name", "category"], how="left")
    category_scores.insert(1, "provider", category_scores["model_name"].map(MODEL_PROVIDER).fillna(""))
    category_scores["rank_within_category"] = category_scores.groupby("category")["category_score"].rank(method="min", ascending=False)
    category_scores.to_csv(out_dir / "13_category_scores.csv", index=False, encoding="utf-8-sig")

    subcategory_scores = item_all.groupby(["model_name", "category", "subcategory"], dropna=False).agg(n_items=("runtime_id", "nunique"), n_benchmarks=("benchmark_id", "nunique"), subcategory_score=("score_norm", "mean"), subcategory_sd=("score_norm", "std")).reset_index()
    subcategory_scores.insert(1, "provider", subcategory_scores["model_name"].map(MODEL_PROVIDER).fillna(""))
    subcategory_scores.to_csv(out_dir / "14_subcategory_scores.csv", index=False, encoding="utf-8-sig")

    qtype_scores = item_all.groupby(["model_name", "question_type"], dropna=False).agg(n_items=("runtime_id", "nunique"), score=("score_norm", "mean"), score_sd=("score_norm", "std")).reset_index()
    qtype_scores.insert(1, "provider", qtype_scores["model_name"].map(MODEL_PROVIDER).fillna(""))
    qtype_scores.to_csv(out_dir / "15_question_type_scores.csv", index=False, encoding="utf-8-sig")

    pair_matrix, pair_long = build_pairwise_win_tables(wide[["Model"] + benchmark_cols].copy(), benchmark_cols)
    pair_matrix.to_csv(out_dir / "16_pairwise_win_matrix.csv", index=False, encoding="utf-8-sig")
    pair_long.to_csv(out_dir / "17_pairwise_win_long.csv", index=False, encoding="utf-8-sig")

    if open_path.exists():
        open_scores = pd.read_csv(open_path, encoding="utf-8-sig")
        dims = [c for c in ["mean_accuracy", "mean_completeness", "mean_clarity", "mean_safety", "mean_overall_raw", "mean_primary_raw", "score_norm", "jury_sd", "jury_parse_success_rate"] if c in open_scores.columns]
        for c in dims:
            open_scores[c] = pd.to_numeric(open_scores[c], errors="coerce")
        agg_spec = {f"{c}_mean": (c, "mean") for c in dims}
        open_dim = open_scores.groupby(["model_name", "benchmark_id", "category", "subcategory"], dropna=False).agg(n_open_items=("runtime_id", "nunique"), **agg_spec).reset_index()
        open_dim.insert(1, "provider", open_dim["model_name"].map(MODEL_PROVIDER).fillna(""))
        open_dim.to_csv(out_dir / "18_open_score_dimensions.csv", index=False, encoding="utf-8-sig")

    if jury_path.exists() and jury_path.stat().st_size > 0 and not jury.empty:
        jury_tendency = jury.copy()
        for c in ["accuracy_score", "completeness_score", "clarity_score", "safety_score", "overall_score", "judge_input_tokens", "judge_output_tokens", "judge_cost_recomputed"]:
            if c in jury_tendency.columns:
                jury_tendency[c] = pd.to_numeric(jury_tendency[c], errors="coerce")
        jury_tendency["judge_parse_success_bool"] = truthy_series(jury_tendency.get("judge_parse_success", pd.Series(False, index=jury_tendency.index)))
        jury_summary = jury_tendency.groupby(["judge_model"], dropna=False).agg(n_scored=("runtime_id", "count"), n_models_under_eval=("model_under_eval", "nunique"), mean_accuracy=("accuracy_score", "mean"), mean_completeness=("completeness_score", "mean"), mean_clarity=("clarity_score", "mean"), mean_safety=("safety_score", "mean"), mean_overall=("overall_score", "mean"), parse_success_rate=("judge_parse_success_bool", "mean"), input_tokens=("judge_input_tokens", "sum"), output_tokens=("judge_output_tokens", "sum"), cost_usd=("judge_cost_recomputed", "sum")).reset_index()
        jury_summary.to_csv(out_dir / "19_jury_model_scoring_tendency_summary.csv", index=False, encoding="utf-8-sig")

        closed_for_combo = item_all[item_all["question_type"] == "closed_qa"][["runtime_id", "instance_id", "benchmark_id", "category", "subcategory", "model_name", "question_type", "score_norm"]].copy()
        combo_rows = compute_subset_leaderboard_rows(closed_for_combo, jury.copy(), benchmark_order)
        if not combo_rows.empty:
            combo_rows.to_csv(out_dir / "20_jury_combination_robustness.csv", index=False, encoding="utf-8-sig")
            jury_robustness_summary = summarize_jury_robustness(combo_rows)
            if not jury_robustness_summary.empty:
                jury_robustness_summary.to_csv(out_dir / "20b_jury_robustness_summary.csv", index=False, encoding="utf-8-sig")

    table_index_rows = [
        {"file": "07_leaderboard.csv", "supports": "Table 1; Fig.3/Fig.4 总体与 benchmark 级模型表现", "notes": "含 Mean win rate、Macro-average、每个 benchmark 分数"},
        {"file": "08_costs.csv", "supports": "Fig.5 成本-性能图", "notes": "按模型和 benchmark 汇总作答成本与 jury 成本"},
        {"file": "22_model_cost_performance_summary.csv", "supports": "Fig.5 成本-性能图和正文成本描述", "notes": "模型级总成本、每 1000 例成本、macro-average 与 win rate"},
        {"file": "13_category_scores.csv", "supports": "Fig.4 类别层面结果", "notes": "模型 × 一级类别表现"},
        {"file": "14_subcategory_scores.csv", "supports": "子类别层面结果段落/补充表", "notes": "模型 × 子类别表现"},
        {"file": "15_question_type_scores.csv", "supports": "封闭题与开放题分层分析", "notes": "closed_qa vs open_task"},
        {"file": "16_pairwise_win_matrix.csv", "supports": "两两模型比较图/补充表", "notes": "主口径：score > opponent；长表另含 tie=0.5 和 tie=win 敏感性列"},
        {"file": "18_open_score_dimensions.csv", "supports": "开放题 accuracy/completeness/clarity/safety 分析", "notes": "安全性作为补充维度，不进入主分数"},
        {"file": "20_jury_combination_robustness.csv", "supports": "Supplementary Table 6：jury 组合稳健性分析长表", "notes": "3 个评分模型的 7 种非空组合"},
        {"file": "20b_jury_robustness_summary.csv", "supports": "Supplementary Table 6：jury 稳健性 compact summary", "notes": "与全体 judge 组合的排名相关、Top-1 和 Top-3 稳定性"},
        {"file": "23_jury_prompt_and_rubric.txt", "supports": "Supplementary Fig.1 和 Methods", "notes": "LLM-jury prompt、评分锚点和主分数定义"},
        {"file": "24_instance_distribution_summary.csv", "supports": "Supplementary Table 2：实例分布", "notes": "240 个实例按 category/subcategory/benchmark/question_type/complexity 汇总"},
        {"file": "12_quality_control_report.csv", "supports": "Methods / Limitations / QC 说明", "notes": "漏跑、解析失败、jury 缺失检查"},
        {"file": "00_benchmark_manifest.csv", "supports": "Supplementary Table 1：benchmark suite 清单", "notes": "12 个 benchmark 的题量、开放/封闭题数量"},
        {"file": "26_paper_figure_table_source_map.csv", "supports": "最终 5Fig+1Table+Extended/Supplementary 的出图/表接续", "notes": "每个计划图表对应的源数据表和代码适配策略"},
        {"file": "27_paper_readiness_report.csv", "supports": "正式写作前完整性检查", "notes": "确认无漏跑、无 API 错误、无解析失败、无 finish_reason=length 截断"},
    ]
    pd.DataFrame(table_index_rows).to_csv(out_dir / "21_analysis_table_index.csv", index=False, encoding="utf-8-sig")

    # 质量控制表：检查是否有漏跑、解析失败或 jury 数不足的问题。正式写论文前应先确认这些问题为 0 或已人工说明。
    qc_rows = []
    expected_total_per_model = len(df)
    expected_closed_per_model = int((df["question_type"] == "closed_qa").sum())
    expected_open_per_model = int((df["question_type"] == "open_task").sum())
    raw_for_qc = raw if not raw.empty else pd.DataFrame()
    closed_for_qc = pd.read_csv(closed_path, encoding="utf-8-sig") if closed_path.exists() else pd.DataFrame()
    open_for_qc = pd.read_csv(open_path, encoding="utf-8-sig") if open_path.exists() else pd.DataFrame()
    jury_for_qc = jury if jury_path.exists() and jury_path.stat().st_size > 0 else pd.DataFrame()
    for m in TEST_MODELS:
        raw_m = raw_for_qc[raw_for_qc["model_name"] == m] if not raw_for_qc.empty else pd.DataFrame()
        closed_m = closed_for_qc[closed_for_qc["model_name"] == m] if not closed_for_qc.empty else pd.DataFrame()
        open_m = open_for_qc[open_for_qc["model_name"] == m] if not open_for_qc.empty else pd.DataFrame()
        jury_m = jury_for_qc[jury_for_qc["model_under_eval"] == m] if not jury_for_qc.empty else pd.DataFrame()
        qc_rows.append({
            "model_name": m,
            "expected_raw_outputs": expected_total_per_model,
            "observed_raw_outputs": int(raw_m[["runtime_id"]].drop_duplicates().shape[0]) if not raw_m.empty else 0,
            "raw_outputs_missing": expected_total_per_model - (int(raw_m[["runtime_id"]].drop_duplicates().shape[0]) if not raw_m.empty else 0),
            "api_error_rows": int(raw_m["error_message"].fillna("").astype(str).ne("").sum()) if not raw_m.empty and "error_message" in raw_m else 0,
            "finish_reason_length_rows": int(raw_m["finish_reason"].fillna("").astype(str).str.lower().eq("length").sum()) if not raw_m.empty and "finish_reason" in raw_m else 0,
            "expected_closed_items": expected_closed_per_model,
            "observed_closed_scores": int(closed_m[["runtime_id"]].drop_duplicates().shape[0]) if not closed_m.empty else 0,
            "closed_parse_or_answer_invalid": int((~closed_m["answer_valid"].astype(str).str.lower().isin(["true", "1", "yes"])).sum()) if not closed_m.empty and "answer_valid" in closed_m else 0,
            "expected_open_items": expected_open_per_model,
            "observed_open_scores": int(open_m[["runtime_id"]].drop_duplicates().shape[0]) if not open_m.empty else 0,
            "expected_jury_rows": expected_open_per_model * len(JUDGE_MODELS),
            "observed_jury_rows": int(jury_m.shape[0]) if not jury_m.empty else 0,
            "jury_rows_missing": expected_open_per_model * len(JUDGE_MODELS) - (int(jury_m.shape[0]) if not jury_m.empty else 0),
            "jury_parse_fail_rows": int((~jury_m["judge_parse_success"].astype(str).str.lower().isin(["true", "1", "yes"])).sum()) if not jury_m.empty and "judge_parse_success" in jury_m else 0,
            "judge_finish_reason_length_rows": int(jury_m["judge_finish_reason"].fillna("").astype(str).str.lower().eq("length").sum()) if not jury_m.empty and "judge_finish_reason" in jury_m else 0,
        })
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(out_dir / "12_quality_control_report.csv", index=False, encoding="utf-8-sig")

    # 即使全程无错误，也写入带表头的空 error log，方便 Supplementary Table 4 和后续脚本稳定读取。
    err_log_path = out_dir / "11_error_retry_log.csv"
    if not err_log_path.exists() or err_log_path.stat().st_size == 0:
        pd.DataFrame(columns=ERROR_FIELDS).to_csv(err_log_path, index=False, encoding="utf-8-sig")

    # 既定论文图表数据来源映射：服务当前 Heliyon 版 5 Figure + 1 Table + 2 Supplementary Figure + 9 Supplementary Table。
    paper_map_rows = [
        {"article_object": "Fig. 1", "planned_content": "Overall TBI-NeuroHELM evaluation framework", "primary_source_files": "09_run_manifest.csv; 10_model_registry.csv; 23_jury_prompt_and_rubric.txt", "plot_code_strategy": "custom schematic / modified reference framework figure"},
        {"article_object": "Fig. 2", "planned_content": "TBI neurology taxonomy and benchmark composition", "primary_source_files": "00_benchmark_manifest.csv; 00_case_id_mapping_enriched.csv; 24_instance_distribution_summary.csv", "plot_code_strategy": "custom taxonomy/composition figure"},
        {"article_object": "Fig. 3", "planned_content": "9 models × 12 benchmarks performance heatmap", "primary_source_files": "06_benchmark_scores_long.csv; 07_leaderboard.csv", "plot_code_strategy": "adapt reference heatmap code"},
        {"article_object": "Fig. 4", "planned_content": "9 models × 4 primary categories performance heatmap", "primary_source_files": "13_category_scores.csv", "plot_code_strategy": "adapt reference category heatmap code"},
        {"article_object": "Fig. 5", "planned_content": "Cost-performance relationship", "primary_source_files": "22_model_cost_performance_summary.csv; 08_costs.csv; 07_leaderboard.csv", "plot_code_strategy": "adapt reference cost-performance code"},
        {"article_object": "Table 1", "planned_content": "Overall leaderboard with strict win rate and macro-average", "primary_source_files": "07_leaderboard.csv; 16_pairwise_win_matrix.csv; 17_pairwise_win_long.csv", "plot_code_strategy": "adapt reference win-rate table code"},
        {"article_object": "Supplementary Figure 1", "planned_content": "20-subcategory model performance", "primary_source_files": "14_subcategory_scores.csv", "plot_code_strategy": "custom / adapted heatmap code"},
        {"article_object": "QC report (not a formal supplementary table)", "planned_content": "Benchmark suite specification", "primary_source_files": "00_benchmark_manifest.csv", "plot_code_strategy": "table export"},
        {"article_object": "Supplementary Table 7", "planned_content": "Model registry and cost settings", "primary_source_files": "10_model_registry.csv; 22_model_cost_performance_summary.csv; model_prices_template.csv", "plot_code_strategy": "table export"},
        {"article_object": "Supplementary Fig. 1", "planned_content": "LLM-jury scoring workflow and rubric", "primary_source_files": "23_jury_prompt_and_rubric.txt; 03_open_jury_scores_raw.csv; 04_open_item_scores.csv", "plot_code_strategy": "custom flowchart"},
        {"article_object": "QC report (not a formal supplementary table)", "planned_content": "Instance distribution summary", "primary_source_files": "24_instance_distribution_summary.csv", "plot_code_strategy": "table export"},
        {"article_object": "QC report (not a formal supplementary table)", "planned_content": "Open-task dimension scores", "primary_source_files": "18_open_score_dimensions.csv", "plot_code_strategy": "table export"},
        {"article_object": "Supplementary Table 6", "planned_content": "LLM-jury robustness", "primary_source_files": "20_jury_combination_robustness.csv; 20b_jury_robustness_summary.csv", "plot_code_strategy": "table export"},
        {"article_object": "QC report (not a formal supplementary table)", "planned_content": "Quality control and run completeness", "primary_source_files": "12_quality_control_report.csv; 11_error_retry_log.csv; 27_paper_readiness_report.csv", "plot_code_strategy": "table export"},
    ]
    pd.DataFrame(paper_map_rows).to_csv(out_dir / "26_paper_figure_table_source_map.csv", index=False, encoding="utf-8-sig")

    required_files = sorted({f.strip() for row in paper_map_rows for f in row["primary_source_files"].split(";") if f.strip() and f.strip() != "27_paper_readiness_report.csv"})
    readiness_rows = []
    def add_readiness(check: str, passed: bool, detail: str) -> None:
        readiness_rows.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})
    for fname in required_files:
        f = out_dir / fname
        add_readiness(f"required_file_exists::{fname}", f.exists() and f.stat().st_size > 0, str(f))
    add_readiness("raw_outputs_complete", bool((qc_df["raw_outputs_missing"] == 0).all()) if not qc_df.empty and "raw_outputs_missing" in qc_df else False, f"missing_sum={int(qc_df['raw_outputs_missing'].sum()) if not qc_df.empty and 'raw_outputs_missing' in qc_df else 'NA'}")
    add_readiness("no_api_error_rows", bool((qc_df["api_error_rows"] == 0).all()) if not qc_df.empty and "api_error_rows" in qc_df else False, f"api_error_sum={int(qc_df['api_error_rows'].sum()) if not qc_df.empty and 'api_error_rows' in qc_df else 'NA'}")
    add_readiness("no_output_truncation_finish_reason_length", bool((qc_df["finish_reason_length_rows"] == 0).all()) if not qc_df.empty and "finish_reason_length_rows" in qc_df else True, f"length_finish_sum={int(qc_df['finish_reason_length_rows'].sum()) if not qc_df.empty and 'finish_reason_length_rows' in qc_df else 0}")
    add_readiness("closed_scores_complete", bool((qc_df["observed_closed_scores"] == qc_df["expected_closed_items"]).all()) if not qc_df.empty and {"observed_closed_scores", "expected_closed_items"}.issubset(qc_df.columns) else False, "closed exact-match rows match expected per model")
    add_readiness("closed_answers_parseable", bool((qc_df["closed_parse_or_answer_invalid"] == 0).all()) if not qc_df.empty and "closed_parse_or_answer_invalid" in qc_df else False, f"invalid_sum={int(qc_df['closed_parse_or_answer_invalid'].sum()) if not qc_df.empty and 'closed_parse_or_answer_invalid' in qc_df else 'NA'}")
    add_readiness("open_scores_complete", bool((qc_df["observed_open_scores"] == qc_df["expected_open_items"]).all()) if not qc_df.empty and {"observed_open_scores", "expected_open_items"}.issubset(qc_df.columns) else False, "open-item aggregated score rows match expected per model")
    add_readiness("jury_rows_complete", bool((qc_df["jury_rows_missing"] == 0).all()) if not qc_df.empty and "jury_rows_missing" in qc_df else False, f"jury_missing_sum={int(qc_df['jury_rows_missing'].sum()) if not qc_df.empty and 'jury_rows_missing' in qc_df else 'NA'}")
    add_readiness("jury_parse_success_complete", bool((qc_df["jury_parse_fail_rows"] == 0).all()) if not qc_df.empty and "jury_parse_fail_rows" in qc_df else False, f"jury_parse_fail_sum={int(qc_df['jury_parse_fail_rows'].sum()) if not qc_df.empty and 'jury_parse_fail_rows' in qc_df else 'NA'}")
    add_readiness("no_jury_output_truncation_finish_reason_length", bool((qc_df["judge_finish_reason_length_rows"] == 0).all()) if not qc_df.empty and "judge_finish_reason_length_rows" in qc_df else True, f"judge_length_finish_sum={int(qc_df['judge_finish_reason_length_rows'].sum()) if not qc_df.empty and 'judge_finish_reason_length_rows' in qc_df else 0}")
    pd.DataFrame(readiness_rows).to_csv(out_dir / "27_paper_readiness_report.csv", index=False, encoding="utf-8-sig")

    # 保存运行清单和模型登记表，方便写 Methods 和复现。
    manifest = pd.DataFrame([{
        "run_id": run_id,
        "script_version": SCRIPT_VERSION,
        "run_date": now_iso(),
        "input_file": str(args.input_csv),
        "mapping_file": str(args.mapping_xlsx),
        "benchmark_file": str(args.benchmark_xlsx),
        "prompt_file": str(args.prompt_txt),
        "system_prompt_hash": sha256_text(system_prompt),
        "input_csv_sha256": sha256_file(Path(args.input_csv)),
        "mapping_xlsx_sha256": sha256_file(Path(args.mapping_xlsx)),
        "benchmark_xlsx_sha256": sha256_file(Path(args.benchmark_xlsx)),
        "prompt_txt_sha256": sha256_file(Path(args.prompt_txt)),
        "evaluation_config_hash": getattr(args, "evaluation_config_hash", ""),
        "n_models": len(TEST_MODELS),
        "n_judges": len(JUDGE_MODELS),
        "n_items": len(df),
        "n_closed": int((df["question_type"] == "closed_qa").sum()),
        "n_open": int((df["question_type"] == "open_task").sum()),
        "temperature": args.temperature,
        "closed_max_tokens": args.closed_max_tokens,
        "open_max_tokens": args.open_max_tokens,
        "jury_max_tokens": args.jury_max_tokens,
        "api_base": args.api_base,
        "workers": args.workers,
        "allow_api_param_fallback": ALLOW_API_PARAM_FALLBACK,
        "api_param_policy": "strict: same temperature and task-type max_tokens are sent to all tested models and judge models unless ALLOW_API_PARAM_FALLBACK is manually changed",
        "tie_policy_for_win_rate": "primary strict win: score>opponent counts as win; tie=0.5 and tie=win sensitivity columns also saved",
        "open_score_normalization": "score_norm=(mean_primary_raw-1)/4; mean_primary_raw=mean(accuracy,completeness,clarity)",
        "api_key_strategy": "separate keys by phase/model; fallback to GLOBAL_APIYI_API_KEY or environment APIYI_API_KEY",
        "output_dir_policy": "Do not mix different prompts, model lists, scoring rules, temperatures or token budgets in the same OUT_DIR; v18 writes 00_evaluation_config.json to enforce this.",
    }])
    manifest.to_csv(out_dir / "09_run_manifest.csv", index=False, encoding="utf-8-sig")

    observed_test_response_models: Dict[str, str] = {}
    if not raw.empty and "api_response_model" in raw.columns:
        tmp_obs = raw.copy()
        tmp_obs["api_response_model"] = tmp_obs["api_response_model"].fillna("").astype(str)
        observed_test_response_models = tmp_obs.groupby("model_name")["api_response_model"].apply(
            lambda x: " | ".join(sorted({v for v in x if v.strip()}))
        ).to_dict()
    observed_judge_response_models: Dict[str, str] = {}
    if not jury.empty and "judge_api_response_model" in jury.columns:
        tmp_jobs = jury.copy()
        tmp_jobs["judge_api_response_model"] = tmp_jobs["judge_api_response_model"].fillna("").astype(str)
        observed_judge_response_models = tmp_jobs.groupby("judge_model")["judge_api_response_model"].apply(
            lambda x: " | ".join(sorted({v for v in x if v.strip()}))
        ).to_dict()

    reg_rows = []
    for m in sorted(set(TEST_MODELS + JUDGE_MODELS)):
        hit = price_df.loc[price_df["model_name"] == m]
        row = hit.iloc[0] if not hit.empty else {}
        in_price = row.get("input_price_per_1m") if not hit.empty else None
        out_price = row.get("output_price_per_1m") if not hit.empty else None
        reg_rows.append({
            "model_name": m,
            "provider": MODEL_PROVIDER.get(m, ""),
            "api_model_id_requested": m,
            "observed_api_response_models_test": observed_test_response_models.get(m, ""),
            "observed_api_response_models_judge": observed_judge_response_models.get(m, ""),
            "used_as_test_model": m in TEST_MODELS,
            "used_as_judge_model": m in JUDGE_MODELS,
            "test_api_key_label": key_label_for_model("test", m) if m in TEST_MODELS else "",
            "judge_api_key_label": key_label_for_model("judge", m) if m in JUDGE_MODELS else "",
            "input_price_per_1m": in_price,
            "output_price_per_1m": out_price,
            "tier_threshold_input_tokens": row.get("tier_threshold_input_tokens") if not hit.empty else None,
            "input_price_over_threshold_per_1m": row.get("input_price_over_threshold_per_1m") if not hit.empty else None,
            "output_price_over_threshold_per_1m": row.get("output_price_over_threshold_per_1m") if not hit.empty else None,
        })
    pd.DataFrame(reg_rows).to_csv(out_dir / "10_model_registry.csv", index=False, encoding="utf-8-sig")

    # 额外保存一个总 Excel，方便人工查看。
    workbook_path = out_dir / "TBI_NeuroHELM_results.xlsx"
    sheet_files = [
        "00_case_id_mapping_enriched.csv",
        "01_model_outputs_raw.csv",
        "02_closed_item_scores.csv",
        "03_open_jury_scores_raw.csv",
        "04_open_item_scores.csv",
        "05_item_scores_all.csv",
        "06_benchmark_scores_long.csv",
        "07_leaderboard.csv",
        "08_costs.csv",
        "08b_costs_by_api_key_label.csv",
        "09_run_manifest.csv",
        "10_model_registry.csv",
        "11_error_retry_log.csv",
        "12_quality_control_report.csv",
        "13_category_scores.csv",
        "14_subcategory_scores.csv",
        "15_question_type_scores.csv",
        "16_pairwise_win_matrix.csv",
        "17_pairwise_win_long.csv",
        "18_open_score_dimensions.csv",
        "19_jury_model_scoring_tendency_summary.csv",
        "20_jury_combination_robustness.csv",
        "20b_jury_robustness_summary.csv",
        "21_analysis_table_index.csv",
        "22_model_cost_performance_summary.csv",
        "24_instance_distribution_summary.csv",
        "26_paper_figure_table_source_map.csv",
        "27_paper_readiness_report.csv",
        "00_preflight_validation.csv",
        "00_benchmark_manifest.csv",
    ]
    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        for fname in sheet_files:
            f = out_dir / fname
            if f.exists() and f.stat().st_size > 0:
                sdf = pd.read_csv(f, encoding="utf-8-sig")
                # Excel 的 sheet 名不能超过 31 个字符。
                sheet = fname.replace(".csv", "")[:31]
                sdf.to_excel(writer, index=False, sheet_name=sheet)
                ws = writer.sheets[sheet]
                ws.freeze_panes(1, 0)
                for i, col in enumerate(sdf.columns):
                    width = min(max(len(str(col)) + 2, 12), 38)
                    ws.set_column(i, i, width)
    print(f"[汇总] 最终结果表已写入：{out_dir}")

# ----------------------------
# smoke 模式：少量真实 API 连通性测试，不写入正式原始结果表
# ----------------------------

def run_api_smoke_test(
    df: pd.DataFrame,
    system_prompt: str,
    clients: Dict[Tuple[str, str], APIYiClient],
    out_dir: Path,
    run_id: str,
    args: argparse.Namespace,
) -> None:
    """用极少量调用验证 9 个被测模型和 3 个 judge 模型的 key、参数和解析逻辑。

    v18.2-progress：增加 smoke 和 all/infer/jury 长耗时阶段的控制台进度打印，不改变请求、评分、汇总和输出文件逻辑。
    """
    closed_candidates = df[df["question_type"] == "closed_qa"]
    open_candidates = df[df["question_type"] == "open_task"]
    if closed_candidates.empty or open_candidates.empty:
        raise RuntimeError("Smoke test requires at least one closed_qa and one open_task item.")
    closed_row = closed_candidates.iloc[0]
    open_row = open_candidates.iloc[0]
    rows: List[Dict[str, Any]] = []
    first_open_response = ""
    first_open_model = ""

    # 进度显示：smoke 一共是 9 个模型 × 2 道题 + 3 个 judge = 21 次 API 调用。
    # 这里只打印到控制台，不改变任何结果表或评测逻辑。
    total_calls = len(TEST_MODELS) * 2 + len(JUDGE_MODELS)
    completed_calls = 0
    smoke_started_at = time.time()

    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hh, rem = divmod(seconds, 3600)
        mm, ss = divmod(rem, 60)
        if hh:
            return f"{hh:d}h{mm:02d}m{ss:02d}s"
        if mm:
            return f"{mm:d}m{ss:02d}s"
        return f"{ss:d}s"

    def _log_start(label: str) -> None:
        print(f"[smoke] 开始 {completed_calls + 1}/{total_calls}: {label}", flush=True)

    def _log_done(label: str, ok: bool, parse_success: Any, finish_reason: str, latency_seconds: float, error_message: str = "") -> None:
        nonlocal completed_calls
        completed_calls += 1
        elapsed = time.time() - smoke_started_at
        avg_per_call = elapsed / completed_calls if completed_calls else 0.0
        eta = avg_per_call * (total_calls - completed_calls)
        pct = completed_calls / total_calls * 100 if total_calls else 100.0
        err_short = f" | error={str(error_message)[:120]}" if error_message else ""
        print(
            f"[smoke] 完成 {completed_calls}/{total_calls} ({pct:.1f}%) | {label}"
            f" | ok={ok} | parse_success={parse_success} | finish_reason={finish_reason or 'NA'}"
            f" | latency={latency_seconds:.1f}s | elapsed={_format_duration(elapsed)} | ETA={_format_duration(eta)}{err_short}",
            flush=True,
        )

    print(
        f"[smoke] 即将开始真实 API 连通性测试：{len(TEST_MODELS)} 个被测模型 × 2 道题 + "
        f"{len(JUDGE_MODELS)} 个 judge = {total_calls} 次 API 调用。",
        flush=True,
    )

    def add_row(**kwargs: Any) -> None:
        base = {
            "run_id": run_id,
            "finished_at": now_iso(),
            "temperature": args.temperature,
            "allow_api_param_fallback": ALLOW_API_PARAM_FALLBACK,
        }
        base.update(kwargs)
        rows.append(base)

    for model in TEST_MODELS:
        client = clients[("test", model)]
        for rr, qtype, max_tok in [(closed_row, "closed_qa", args.closed_max_tokens), (open_row, "open_task", args.open_max_tokens)]:
            label = f"test_model | {model} | {qtype} | {rr.get('runtime_id', '')}"
            _log_start(label)
            t0 = time.time()
            try:
                res = client.chat_completion(model=model, messages=build_test_messages(system_prompt, rr), temperature=args.temperature, max_tokens=max_tok)
                content = res.get("content", "") or ""
                parsed_answer = ""
                parse_success = None
                if qtype == "closed_qa":
                    parsed_answer = extract_answer_letters(content, valid_options=option_letters_from_prompt(str(rr.get("prompt", ""))), reference_mode=False)
                    parse_success = bool(parsed_answer)
                else:
                    parse_success = bool(content.strip())
                    if res.get("ok") and content.strip() and not first_open_response:
                        first_open_response = content
                        first_open_model = model
                latency_seconds = round(time.time() - t0, 3)
                add_row(
                    phase="test_model",
                    model_name=model,
                    judge_model="",
                    runtime_id=rr.get("runtime_id", ""),
                    question_type=qtype,
                    requested_max_tokens=max_tok,
                    ok=bool(res.get("ok")),
                    parse_success=parse_success,
                    parsed_answer=parsed_answer,
                    api_response_model=res.get("response_model", ""),
                    finish_reason=res.get("finish_reason", ""),
                    input_tokens=res.get("prompt_tokens"),
                    output_tokens=res.get("completion_tokens"),
                    total_tokens=res.get("total_tokens"),
                    latency_seconds=latency_seconds,
                    error_message=res.get("error", ""),
                    response_excerpt=str(content)[:500],
                )
                _log_done(label, bool(res.get("ok")), parse_success, str(res.get("finish_reason", "")), latency_seconds, str(res.get("error", "")))
            except Exception as e:
                latency_seconds = round(time.time() - t0, 3)
                add_row(
                    phase="test_model",
                    model_name=model,
                    judge_model="",
                    runtime_id=rr.get("runtime_id", ""),
                    question_type=qtype,
                    requested_max_tokens=max_tok,
                    ok=False,
                    parse_success=False,
                    parsed_answer="",
                    api_response_model="",
                    finish_reason="",
                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=None,
                    latency_seconds=latency_seconds,
                    error_message=repr(e),
                    response_excerpt="",
                )
                _log_done(label, False, False, "", latency_seconds, repr(e))

    candidate_response = first_open_response or "Smoke-test placeholder: no usable tested-model open-task response was available."
    for judge in JUDGE_MODELS:
        client = clients[("judge", judge)]
        label = f"judge_model | {judge} | score {first_open_model or 'placeholder'} | {open_row.get('runtime_id', '')}"
        _log_start(label)
        t0 = time.time()
        try:
            jury_user_prompt = build_jury_prompt(open_row, candidate_response)
            res = client.chat_completion(
                model=judge,
                messages=[{"role": "system", "content": JURY_SYSTEM_PROMPT}, {"role": "user", "content": jury_user_prompt}],
                temperature=0.0,
                max_tokens=args.jury_max_tokens,
            )
            content = res.get("content", "") or ""
            parsed_obj, parse_ok = parse_jury_score_object(content)
            latency_seconds = round(time.time() - t0, 3)
            add_row(
                phase="judge_model",
                model_name=first_open_model,
                judge_model=judge,
                runtime_id=open_row.get("runtime_id", ""),
                question_type="open_task",
                requested_max_tokens=args.jury_max_tokens,
                ok=bool(res.get("ok")),
                parse_success=bool(parse_ok),
                parsed_answer="",
                api_response_model=res.get("response_model", ""),
                finish_reason=res.get("finish_reason", ""),
                input_tokens=res.get("prompt_tokens"),
                output_tokens=res.get("completion_tokens"),
                total_tokens=res.get("total_tokens"),
                latency_seconds=latency_seconds,
                error_message=res.get("error", ""),
                response_excerpt=str(content)[:500],
                accuracy_score=parsed_obj.get("accuracy_score"),
                completeness_score=parsed_obj.get("completeness_score"),
                clarity_score=parsed_obj.get("clarity_score"),
                safety_score=parsed_obj.get("safety_score"),
                overall_score=parsed_obj.get("overall_score"),
            )
            _log_done(label, bool(res.get("ok")), bool(parse_ok), str(res.get("finish_reason", "")), latency_seconds, str(res.get("error", "")))
        except Exception as e:
            latency_seconds = round(time.time() - t0, 3)
            add_row(
                phase="judge_model",
                model_name=first_open_model,
                judge_model=judge,
                runtime_id=open_row.get("runtime_id", ""),
                question_type="open_task",
                requested_max_tokens=args.jury_max_tokens,
                ok=False,
                parse_success=False,
                parsed_answer="",
                api_response_model="",
                finish_reason="",
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                latency_seconds=latency_seconds,
                error_message=repr(e),
                response_excerpt="",
            )
            _log_done(label, False, False, "", latency_seconds, repr(e))

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "25_api_smoke_test_report.csv", index=False, encoding="utf-8-sig")
    print(f"[smoke] 已完成少量真实 API 测试，报告见：{out_dir / '25_api_smoke_test_report.csv'}", flush=True)


# ----------------------------
# 命令行参数：也可以不管这里，直接改上面的“用户需要修改的参数区”
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TBI-NeuroHELM benchmark evaluation workflow.")
    parser.add_argument("--input_csv", default=INPUT_CSV)
    parser.add_argument("--mapping_xlsx", default=MAPPING_XLSX)
    parser.add_argument("--benchmark_xlsx", default=BENCHMARK_XLSX)
    parser.add_argument("--prompt_txt", default=PROMPT_TXT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--api_base", default=API_BASE)
    parser.add_argument("--api_key", default=GLOBAL_APIYI_API_KEY or os.getenv("APIYI_API_KEY", ""), help="兜底用总 API易 key；如果模型专属 key 为空，会使用这个 key。")
    parser.add_argument("--price_file", default=PRICE_FILE, help="可选价格表 CSV，字段：model_name,input_price_per_1m,output_price_per_1m")
    parser.add_argument("--mode", choices=["all", "infer", "closed", "jury", "aggregate", "smoke", "preflight"], default=MODE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--closed_max_tokens", type=int, default=CLOSED_MAX_TOKENS)
    parser.add_argument("--open_max_tokens", type=int, default=OPEN_MAX_TOKENS)
    parser.add_argument("--jury_max_tokens", type=int, default=JURY_MAX_TOKENS)
    parser.add_argument("--force", action="store_true", default=FORCE_RERUN, help="强制重跑；已有 CSV 不会自动删除，但会重新请求 API，容易重复花钱。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_csv = resolve_path(args.input_csv, script_dir)
    mapping_xlsx = resolve_path(args.mapping_xlsx, script_dir)
    benchmark_xlsx = resolve_path(args.benchmark_xlsx, script_dir)
    prompt_txt = resolve_path(args.prompt_txt, script_dir)
    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    args.input_csv = str(input_csv)
    args.mapping_xlsx = str(mapping_xlsx)
    args.benchmark_xlsx = str(benchmark_xlsx)
    args.prompt_txt = str(prompt_txt)

    system_prompt = read_text(prompt_txt)
    df = load_and_prepare_data(input_csv, mapping_xlsx, benchmark_xlsx, out_dir)

    price_file = Path(args.price_file) if args.price_file else None
    if price_file and not price_file.exists():
        print(f"[警告] 找不到价格表：{price_file}；将使用脚本内置默认价格。")
        price_file = None
    price_df = load_or_create_price_table(out_dir, price_file)
    evaluation_config_payload = build_evaluation_config_payload(system_prompt, args, input_csv, mapping_xlsx, benchmark_xlsx, prompt_txt)
    evaluation_config_hash = write_and_validate_evaluation_config(out_dir, evaluation_config_payload, args.mode, args.force)
    args.evaluation_config_hash = evaluation_config_hash
    write_preflight_tables(df, out_dir, price_df, system_prompt, args, evaluation_config_hash)
    if args.mode == "preflight":
        print(f"[preflight] 已完成输入检查和配置指纹写入；未调用 API。结果见：{out_dir}")
        return

    run_id = f"tbi_neurohelm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    clients = None
    if args.mode in ["all", "infer", "jury", "smoke"]:
        # 根据 TEST_MODEL_API_KEYS / JUDGE_MODEL_API_KEYS 为每个模型创建独立 client。
        # 如果某个专属 key 为空，会使用 args.api_key 作为兜底。
        clients = build_api_clients(args.api_base, args.timeout, args.max_retries, fallback_key=args.api_key)

    if args.mode == "smoke":
        assert clients is not None
        run_api_smoke_test(df, system_prompt, clients, out_dir, run_id, args)
        return

    if args.mode in ["all", "infer"]:
        assert clients is not None
        run_inference(df, system_prompt, clients, out_dir, run_id, args.workers, args.force, args.temperature, args.closed_max_tokens, args.open_max_tokens)

    if args.mode in ["all", "closed"]:
        score_closed_items(df, out_dir)
        print("[封闭题判分] 已生成 02_closed_item_scores.csv")

    if args.mode in ["all", "jury"]:
        assert clients is not None
        run_jury_scoring(df, clients, out_dir, run_id, args.workers, args.force, price_df, args.jury_max_tokens)
        aggregate_open_scores(out_dir)
        print("[开放题 jury] 已生成 03_open_jury_scores_raw.csv 和 04_open_item_scores.csv")

    if args.mode in ["all", "aggregate"]:
        aggregate_all_tables(df, out_dir, price_df, run_id, system_prompt, args)

    print("全部完成。")


if __name__ == "__main__":
    main()
