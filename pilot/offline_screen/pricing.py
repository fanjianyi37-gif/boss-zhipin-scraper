"""deepseek-flash 官方价目换算。

数字抄自 2026-11-10 读取的官方文档：
https://api-docs.deepseek.com/quick_start/pricing/

合批一次请求不能把 usage 拆成单条岗位的实测费用。逐条实测费用固定为未知。
只有缓存命中、缓存未命中和输出 token 都是整数，并且时段能够确定时，才计算整批费用。
时段不能确定时，整批实测费用仍是未知，另外给出标注过的估算。
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing/"
PRICE_CHECKED_ON = "2026-11-10"
PRICE_MODEL = "deepseek-flash"
UNIT = Decimal(1000000)
CENT = Decimal("0.00000001")

# 每 100 万 token 的美元单价。高峰是周一到周五 UTC 01:00-04:00 与 06:00-10:00，
# 官方写明不含中国法定节假日。本模块没有节假日表，所以落在这窗口内的工作日不定价。
RATES = {
    "cache_hit_input": {"off_peak": Decimal("0.003"), "peak": Decimal("0.006")},
    "cache_miss_input": {"off_peak": Decimal("0.15"), "peak": Decimal("0.3")},
    "output": {"off_peak": Decimal("0.6"), "peak": Decimal("1.2")},
}


def price_band(moment):
    """返回 off_peak、unresolved_holiday 或 unknown_timezone。不猜测节假日。"""
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        return "unknown_timezone"
    moment = moment.astimezone(timezone.utc)
    if moment.weekday() >= 5:
        return "off_peak"
    minutes = moment.hour * 60 + moment.minute
    in_peak_clock = (60 <= minutes < 4 * 60) or (6 * 60 <= minutes < 10 * 60)
    if in_peak_clock:
        return "unresolved_holiday"
    return "off_peak"


def _money(value):
    return str(value.quantize(CENT, rounding=ROUND_HALF_UP))


def _int_token(usage, key):
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _scenario(hit, miss, output_tokens, job_count, band):
    batch = (
        Decimal(hit) * RATES["cache_hit_input"][band]
        + Decimal(miss) * RATES["cache_miss_input"][band]
        + Decimal(output_tokens) * RATES["output"][band]
    ) / UNIT
    amortized = None
    if job_count:
        amortized = _money(batch / Decimal(job_count))
    return {
        "label": "estimate_not_measurement",
        "band": band,
        "batch_usd": _money(batch),
        "amortized_per_item_usd": amortized,
        "denominator_jobs": job_count,
    }


def cost_report(usage, job_count, moment=None):
    report = {
        "price_source": PRICE_SOURCE,
        "price_checked_on": PRICE_CHECKED_ON,
        "currency": "USD",
        "model": PRICE_MODEL,
        "per_item_measured_cost": "unknown",
        "per_item_reason": "一次合批请求的 usage 不能拆成单条岗位的实测费用",
        "batch_measured_cost": "unknown",
        "batch_measured_reason": "",
        "amortized_estimate": None,
    }
    if not isinstance(usage, dict):
        report["batch_measured_reason"] = "没有 usage，离线或请求未发生"
        return report
    hit = _int_token(usage, "prompt_cache_hit_tokens")
    miss = _int_token(usage, "prompt_cache_miss_tokens")
    output_tokens = _int_token(usage, "completion_tokens")
    if hit is None or miss is None or output_tokens is None:
        report["batch_measured_reason"] = (
            "usage 缺少缓存命中、缓存未命中或输出 token 的非负整数，不能选定单价"
        )
        return report
    band = price_band(moment)
    if band == "off_peak":
        chosen = _scenario(hit, miss, output_tokens, job_count, "off_peak")
        report["batch_measured_cost"] = chosen["batch_usd"]
        report["batch_measured_reason"] = "按官方价目和 usage 的缓存拆分计算的整批费用"
        report["amortized_estimate"] = chosen
        return report
    report["batch_measured_reason"] = (
        "时段不能确定，或时刻没有时区。高峰窗口可能碰上中国法定节假日，因此整批实测费用记为未知"
    )
    report["amortized_estimate"] = {
        "label": "estimate_not_measurement",
        "note": "两档都是估算，不是逐条或整批实测",
        "off_peak": _scenario(hit, miss, output_tokens, job_count, "off_peak"),
        "peak": _scenario(hit, miss, output_tokens, job_count, "peak"),
    }
    return report
