#!/usr/bin/env python3
"""隔离的岗位筛选试验运行器。

默认不联网。真实生成请求必须同时满足：命令行显式 --live、偏好已冻结、
人工期望已冻结，并且进程环境里有 DEEPSEEK_API_KEY。即便如此，整次试验
最多一次 https://api.deepseek.com/chat/completions，模型固定 deepseek-flash，
自动重试为 0。超时或校验失败后不再请求。

reason 只是筛选短评。本运行器不产生摘要，也不放宽 scripts/deepseek_screen.py
里的校验。
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

from pricing import cost_report

OFFICIAL_URL = "https://api.deepseek.com/chat/completions"
OFFICIAL_MODEL = "deepseek-flash"
MAX_GENERATION_REQUESTS = 1
SCHEMA_NAME = "deepseek_screen.items.v1"
SCHEMA_FIELDS = ("id", "group", "reason")
MODEL_STRUCTURE_STAGES = {
    "non_json",
    "truncated",
    "root_schema",
    "row_schema",
    "id_type",
    "unknown_id",
    "duplicate_id",
    "missing_id",
    "group_type",
    "group",
    "reason_type",
    "reason_length",
    "reason_format",
    "generated_link",
    "ungrounded_salary",
    "ungrounded_degree",
    "ungrounded_place",
    "api_shape",
}
_ADAPTERS = {}


def load_adapter(bundle_dir):
    """只加载包内这一份适配器，不搜索主目录或其他路径。"""
    path = os.path.abspath(os.path.join(bundle_dir, "deepseek_screen.py"))
    cached = _ADAPTERS.get(path)
    if cached is not None:
        return cached
    if not os.path.isfile(path):
        raise FileNotFoundError("bundle is missing deepseek_screen.py")
    spec = importlib.util.spec_from_file_location("deepseek_screen_bundle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "API_URL", None) != OFFICIAL_URL:
        raise RuntimeError("adapter endpoint is not the official chat completions URL")
    if getattr(module, "DEFAULT_MODEL", None) != OFFICIAL_MODEL:
        raise RuntimeError("adapter default model is not deepseek-flash")
    for name in ("validate_items", "run_screen", "normalize_records", "GROUPS"):
        if not hasattr(module, name):
            raise RuntimeError(f"adapter is missing {name}")
    _ADAPTERS[path] = module
    return module


class OneShot:
    """包住传输函数。第二次调用直接拒绝，并且不会再进入内层。"""

    def __init__(self, inner, adapter):
        self.inner = inner
        self.adapter = adapter
        self.calls = 0

    def __call__(self, body, timeout):
        if self.calls >= MAX_GENERATION_REQUESTS:
            raise self.adapter.ModelOutputError("second_request_refused")
        if not isinstance(body, dict) or body.get("model") != OFFICIAL_MODEL:
            raise self.adapter.ModelOutputError("model_refused")
        if body.get("stream") is not False:
            raise self.adapter.ModelOutputError("stream_refused")
        try:
            result = self.inner(body, timeout)
        except self.adapter.MissingKeyError:
            raise
        except Exception:
            self.calls += 1
            raise
        self.calls += 1
        return result


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_jobs(path):
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        return payload["jobs"]
    raise ValueError("fixture must be a list or an object with jobs")


def is_frozen(document):
    return (
        isinstance(document, dict)
        and document.get("status") == "frozen"
        and isinstance(document.get("frozen_at"), str)
        and bool(document.get("frozen_at").strip())
    )


def expectation_problem(document, adapter):
    if not is_frozen(document):
        return "expectations_not_frozen"
    items = document.get("items")
    if not isinstance(items, list) or not items:
        return "expectations_invalid"
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            return "expectations_invalid"
        ident = item.get("id")
        group = item.get("expected_group")
        if not isinstance(ident, str) or not ident or ident in seen:
            return "expectations_invalid"
        if group not in adapter.GROUPS:
            return "expectations_invalid"
        seen.add(ident)
    return ""


def preference_problem(document):
    if not is_frozen(document):
        return "preference_not_frozen"
    text = document.get("text")
    if not isinstance(text, str) or not text.strip():
        return "preference_not_frozen"
    return ""


def sample_meta(meta, job_count):
    kind = "unlabeled"
    real = False
    note = "没有样本说明。未标成真实 JD。"
    if isinstance(meta, dict):
        if meta.get("kind") == "public_fixture":
            kind = "public_fixture"
        note = str(meta.get("note") or note)
        real = False
    return {
        "kind": kind,
        "count": job_count,
        "is_real_jd_sample": real,
        "note": note,
    }


def source_text(record):
    return {
        "id": record["id"],
        "company": record["company"],
        "title": record["title"],
        "salary": record["salary"],
        "location": record["location"],
        "education": record["education"],
        "experience": record["experience"],
        "history": record["history"],
        "detail_status": record["detail_status"],
        "flags": list(record["flags"]),
        "description_excerpt": record["description_excerpt"],
        "description_chars": len(record["description_full"]),
        "tags_list": record.get("tags_list") or "",
        "requirements_preserved": bool(record.get("requirements_preserved", True)),
        "url": record["url"],
    }


def structural_result(job_count, outcome):
    if outcome is None:
        return {
            "status": "not_run",
            "passed_items": None,
            "denominator_jobs": job_count,
            "rate": None,
            "failure_stage": None,
            "note": "未调用模型，没有可校验的输出",
        }
    if outcome.status == "ok":
        rate = 1 if job_count else None
        return {
            "status": "passed",
            "passed_items": job_count,
            "denominator_jobs": job_count,
            "rate": rate,
            "failure_stage": None,
            "note": "整批通过既有校验。失败时整批作废，不会出现部分通过。",
        }
    if outcome.stage in MODEL_STRUCTURE_STAGES:
        rate = 0 if job_count else None
        return {
            "status": "failed",
            "passed_items": 0,
            "denominator_jobs": job_count,
            "rate": rate,
            "failure_stage": outcome.stage,
            "note": "服务返回了模型输出，但整批没通过既有校验。分母是输入岗位数。",
        }
    return {
        "status": "not_run",
        "passed_items": None,
        "denominator_jobs": job_count,
        "rate": None,
        "failure_stage": outcome.stage,
        "note": "传输失败或服务没有返回可校验的模型输出。结构通过率不适用，不是 0。",
    }


def score_expectations(records, model_items, expectations, adapter):
    if not is_frozen(expectations):
        return {
            "status": "not_scored",
            "reason": "人工期望尚未冻结",
            "matches": None,
            "denominator": None,
            "rate": None,
            "missing_expectation_ids": [],
            "expectation_not_in_sample_ids": [],
        }
    if expectation_problem(expectations, adapter):
        return {
            "status": "not_scored",
            "reason": "人工期望文件不可用",
            "matches": None,
            "denominator": None,
            "rate": None,
            "missing_expectation_ids": [],
            "expectation_not_in_sample_ids": [],
        }
    if not model_items:
        return {
            "status": "not_scored",
            "reason": "没有通过校验的模型输出，不能和人工期望比",
            "matches": None,
            "denominator": None,
            "rate": None,
            "missing_expectation_ids": [],
            "expectation_not_in_sample_ids": [],
        }
    expected = {item["id"]: item["expected_group"] for item in expectations["items"]}
    sample_ids = [record["id"] for record in records]
    sample_set = set(sample_ids)
    comparable = [ident for ident in sample_ids if ident in expected]
    missing = [ident for ident in sample_ids if ident not in expected]
    extra = [ident for ident in expected if ident not in sample_set]
    got = {item["id"]: item["group"] for item in model_items}
    matches = sum(1 for ident in comparable if got.get(ident) == expected[ident])
    denominator = len(comparable)
    rate = (matches / denominator) if denominator else None
    return {
        "status": "scored",
        "reason": "只比较同时出现在样本和已冻结期望里的 id",
        "matches": matches,
        "denominator": denominator,
        "rate": rate,
        "missing_expectation_ids": missing,
        "expectation_not_in_sample_ids": extra,
    }


def item_rows(records, outcome):
    outputs = {}
    if outcome is not None and outcome.items:
        outputs = {item["id"]: item for item in outcome.items}
    rows = []
    for record in records:
        output = outputs.get(record["id"])
        if outcome is None:
            validation = "not_called"
        elif output:
            validation = "accepted"
        else:
            validation = outcome.stage
        rows.append({
            "id": record["id"],
            "source_text": source_text(record),
            "output": output,
            "validation": validation,
        })
    return rows


def blank_http():
    return {
        "generation_http_requests": 0,
        "automatic_retries": 0,
        "dispatch_attempted": False,
        "service_received": False,
        "status": None,
        "finish_reason": None,
        "stage": "not_called",
        "fallback": None,
        "error_type": None,
        "error_reason": None,
    }


def http_from_outcome(outcome, request_count):
    return {
        "generation_http_requests": request_count,
        "automatic_retries": 0,
        "dispatch_attempted": request_count > 0,
        "service_received": bool(getattr(outcome, "service_received", False)),
        "status": outcome.http_status,
        "finish_reason": outcome.finish_reason,
        "stage": outcome.stage,
        "fallback": outcome.status != "ok",
        "error_type": getattr(outcome, "error_type", None),
        "error_reason": getattr(outcome, "error_reason", None),
        "prompt_tokens": outcome.prompt_tokens,
        "completion_tokens": outcome.completion_tokens,
    }


def usage_from_outcome(outcome):
    if outcome is None:
        return None
    usage = getattr(outcome, "usage", None)
    if isinstance(usage, dict) and usage:
        return dict(usage)
    if outcome.prompt_tokens is None and outcome.completion_tokens is None:
        return None
    return {
        "prompt_tokens": outcome.prompt_tokens,
        "completion_tokens": outcome.completion_tokens,
    }


def outbound_artifact(adapter, records, preference_text, model):
    """脱敏后的出站消息。不含 Authorization 或密钥。"""
    if any(not record.get("requirements_preserved", True) for record in records):
        return {
            "captured": False,
            "reason": "input_requirements_truncated",
            "messages": None,
            "rules_sha256": None,
            "user_sha256": None,
        }
    body = adapter.build_request(records, preference_text or "", model)
    messages = []
    for message in body["messages"]:
        content = message.get("content") if isinstance(message, dict) else ""
        if not isinstance(content, str):
            content = ""
        messages.append({
            "role": message.get("role") if isinstance(message, dict) else "",
            "content": adapter._redact_text(content),
        })
    return {
        "captured": True,
        "model": body["model"],
        "stream": body["stream"],
        "max_tokens": body["max_tokens"],
        "thinking": body["thinking"],
        "messages": messages,
        "rules_sha256": hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest(),
        "user_sha256": hashlib.sha256(messages[1]["content"].encode("utf-8")).hexdigest(),
    }


def run_trial(
    bundle_dir,
    jobs,
    *,
    sample=None,
    expectations=None,
    preference=None,
    live=False,
    model=OFFICIAL_MODEL,
    transport=None,
    moment=None,
):
    adapter = load_adapter(bundle_dir)
    records, problem = adapter.normalize_records(jobs)
    job_count = len(records) if not problem else len(jobs) if isinstance(jobs, list) else 0
    meta = sample_meta(sample, job_count)
    report = {
        "mode": "live" if live else "offline",
        "trial_accepted": False,
        "sample": meta,
        "schema": {
            "name": SCHEMA_NAME,
            "model_fields": list(SCHEMA_FIELDS),
            "reason_role": "screening_rationale",
            "summary": None,
            "summary_role": "not_produced_by_this_trial",
            "note": "reason 是筛选短评。岗位摘要仍由本地 job_summary 产生，本试验不改那条路径，也不把摘要塞进模型 schema。",
        },
        "limits": {
            "max_generation_http_requests": MAX_GENERATION_REQUESTS,
            "automatic_retries": 0,
            "timeout_stops": True,
            "repair_retry": False,
            "model": OFFICIAL_MODEL,
            "endpoint": OFFICIAL_URL,
        },
        "http": blank_http(),
        "usage": None,
        "cost": cost_report(None, job_count, moment),
        "structural": structural_result(job_count, None),
        "human_expectation": score_expectations(records, None, expectations or {}, adapter),
        "items": item_rows(records, None) if not problem else [],
    }
    if problem:
        report["http"]["stage"] = problem
        report["structural"] = structural_result(job_count, None)
        report["structural"]["failure_stage"] = problem
        report["structural"]["status"] = "not_run"
        return report
    preference_text = ""
    if isinstance(preference, dict) and isinstance(preference.get("text"), str):
        preference_text = preference["text"]
    report["outbound"] = outbound_artifact(adapter, records, preference_text, model)
    if model != OFFICIAL_MODEL:
        report["http"]["stage"] = "model_refused"
        return report
    if not report["outbound"]["captured"]:
        report["http"]["stage"] = "input_requirements_truncated"
        report["structural"]["failure_stage"] = "input_requirements_truncated"
        return report
    if not live:
        report["http"]["stage"] = "offline_not_called"
        return report

    gate = expectation_problem(expectations or {}, adapter)
    if gate:
        report["http"]["stage"] = gate
        return report
    pref_gate = preference_problem(preference or {})
    if pref_gate:
        report["http"]["stage"] = pref_gate
        return report

    oneshot = OneShot(transport or adapter.default_transport, adapter)
    outcome = adapter.run_screen(
        jobs,
        preferences=preference["text"].strip(),
        model=OFFICIAL_MODEL,
        max_attempts=1,
        pause=0,
        transport=oneshot,
    )
    report["http"] = http_from_outcome(outcome, oneshot.calls)
    report["usage"] = usage_from_outcome(outcome)
    report["cost"] = cost_report(report["usage"], job_count, moment)
    report["structural"] = structural_result(job_count, outcome)
    report["human_expectation"] = score_expectations(
        records, outcome.items, expectations, adapter
    )
    report["items"] = item_rows(records, outcome)
    report["response_diagnostics"] = getattr(outcome, "diagnostics", None)
    human = report["human_expectation"]
    report["trial_accepted"] = bool(
        outcome.status == "ok"
        and human.get("status") == "scored"
        and human.get("denominator")
        and human.get("matches") == human.get("denominator")
    )
    return report


def _sent_dir(path):
    name = os.path.basename(os.path.normpath(path)).lower()
    return name == "sent" or name.startswith("sent.") or name.startswith("sent_") or name.startswith(".sent")


def write_report(output_dir, report):
    if _sent_dir(output_dir):
        raise ValueError("refusing a sent marker output directory")
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, "trial_report.json")
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    temporary = target + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, target)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description="离线优先的岗位筛选小样本运行器。默认不联网。")
    parser.add_argument("--bundle-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--fixture-meta", default=None)
    parser.add_argument("--expectations", required=True)
    parser.add_argument("--preference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=OFFICIAL_MODEL)
    args = parser.parse_args(argv)
    try:
        jobs = load_jobs(args.fixture)
        expectations = read_json(args.expectations)
        preference = read_json(args.preference)
        meta = read_json(args.fixture_meta) if args.fixture_meta else None
        report = run_trial(
            args.bundle_dir,
            jobs,
            sample=meta,
            expectations=expectations,
            preference=preference,
            live=args.live,
            model=args.model,
        )
        path = write_report(args.output_dir, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"trial_report_error {type(exc).__name__}", file=sys.stderr)
        return 2
    printable = {
        "mode": report["mode"],
        "stage": report["http"]["stage"],
        "generation_http_requests": report["http"]["generation_http_requests"],
        "trial_accepted": report["trial_accepted"],
        "report": path,
    }
    print(json.dumps(printable, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
