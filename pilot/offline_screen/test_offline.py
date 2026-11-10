#!/usr/bin/env python3
"""在隔离目录里证明默认不联网，并且真实路径最多一次生成请求。"""

import json
import os
import socket
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from unittest import mock

import pricing
import runner


HERE = os.path.dirname(os.path.abspath(__file__))
DECOY = "DECOY-KEY-DO-NOT-USE"


def setUpModule():
    if not os.path.isfile(os.path.join(HERE, "deepseek_screen.py")):
        raise unittest.SkipTest("offline tests run inside the built bundle")


def frozen_expectations(ids):
    return {
        "status": "frozen",
        "frozen_at": "2026-11-10T00:00:00Z",
        "sample_id": "unit-test-only",
        "note": "只给离线测试用，不是真实样本的人工期望。",
        "items": [{"id": ident, "expected_group": "观察"} for ident in ids],
    }


def frozen_preference():
    return {
        "status": "frozen",
        "frozen_at": "2026-11-10T00:00:00Z",
        "text": "离线测试偏好。不是生产策略。",
        "note": "unit test",
    }


def load_public_jobs():
    with open(os.path.join(HERE, "public_fixture.json"), encoding="utf-8") as handle:
        jobs = json.load(handle)
    with open(os.path.join(HERE, "public_fixture_meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    with open(os.path.join(HERE, "expectations.json"), encoding="utf-8") as handle:
        expectations = json.load(handle)
    with open(os.path.join(HERE, "preference.json"), encoding="utf-8") as handle:
        preference = json.load(handle)
    return jobs, meta, expectations, preference


def valid_content(jobs):
    reasons = {
        "j1": "仓前，标题含视频方向。",
        "j2": "正文写了硕士，与列表本科冲突，先观察。",
        "j3": "仓前，正文是图片质量评测。",
        "j4": "正文要求北京，与滨江冲突。",
    }
    groups = ["优先", "观察", "观察", "不推荐"]
    return json.dumps({
        "items": [
            {"id": job["id"], "group": group, "reason": reasons[job["id"]]}
            for job, group in zip(jobs, groups)
        ]
    }, ensure_ascii=False)


def completion(content):
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8},
    }


class _NetworkGuard(unittest.TestCase):
    def setUp(self):
        self._connect = socket.socket.connect

        def blocked(sock, *args, **kwargs):
            raise AssertionError("network connect was attempted")

        socket.socket.connect = blocked
        self.addCleanup(setattr, socket.socket, "connect", self._connect)


class OfflineDefaultTests(_NetworkGuard):
    def test_public_fixture_is_not_a_real_sample_and_does_not_call_out(self):
        jobs, meta, expectations, preference = load_public_jobs()
        self.assertEqual(len(jobs), 4)
        self.assertFalse(meta["is_real_jd_sample"])
        self.assertEqual(expectations["status"], "not_frozen")
        self.assertEqual(preference["text"], "")
        report = runner.run_trial(
            HERE,
            jobs,
            sample=meta,
            expectations=expectations,
            preference=preference,
            live=False,
        )
        self.assertEqual(report["mode"], "offline")
        self.assertEqual(report["http"]["generation_http_requests"], 0)
        self.assertEqual(report["http"]["automatic_retries"], 0)
        self.assertEqual(report["http"]["stage"], "offline_not_called")
        self.assertFalse(report["trial_accepted"])
        self.assertIsNone(report["schema"]["summary"])
        self.assertEqual(report["schema"]["model_fields"], ["id", "group", "reason"])
        self.assertEqual(report["schema"]["reason_role"], "screening_rationale")
        self.assertEqual(report["sample"]["kind"], "public_fixture")
        self.assertFalse(report["sample"]["is_real_jd_sample"])
        self.assertEqual(report["sample"]["count"], 4)
        self.assertEqual(report["structural"]["status"], "not_run")
        self.assertEqual(report["structural"]["denominator_jobs"], 4)
        self.assertIsNone(report["structural"]["rate"])
        self.assertEqual(report["human_expectation"]["status"], "not_scored")
        self.assertIsNone(report["human_expectation"]["denominator"])
        self.assertEqual(report["cost"]["per_item_measured_cost"], "unknown")
        self.assertEqual(report["cost"]["batch_measured_cost"], "unknown")
        self.assertIsNone(report["cost"]["amortized_estimate"])
        self.assertEqual([row["id"] for row in report["items"]], ["j1", "j2", "j3", "j4"])
        self.assertEqual(report["items"][0]["validation"], "not_called")
        self.assertIsNone(report["items"][0]["output"])
        self.assertIn("8-12K", report["items"][0]["source_text"]["salary"])

    def test_live_flag_without_frozen_files_makes_no_request(self):
        jobs, meta, expectations, preference = load_public_jobs()
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": DECOY}, clear=False):
            report = runner.run_trial(
                HERE,
                jobs,
                sample=meta,
                expectations=expectations,
                preference=preference,
                live=True,
            )
        self.assertEqual(report["http"]["generation_http_requests"], 0)
        self.assertEqual(report["http"]["stage"], "expectations_not_frozen")
        self.assertNotIn(DECOY, json.dumps(report, ensure_ascii=False))

    def test_cli_default_writes_report_without_reading_decoy_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            decoy_home = os.path.join(tmp, "home")
            os.makedirs(decoy_home)
            with open(os.path.join(decoy_home, ".env"), "w", encoding="utf-8") as handle:
                handle.write(f"DEEPSEEK_API_KEY={DECOY}\n")
            with open(os.path.join(tmp, ".env"), "w", encoding="utf-8") as handle:
                handle.write(f"DEEPSEEK_API_KEY={DECOY}\n")
            output = os.path.join(tmp, "out")
            env = os.environ.copy()
            env.pop("DEEPSEEK_API_KEY", None)
            env["HOME"] = decoy_home
            with mock.patch.dict(os.environ, env, clear=True):
                code = runner.main([
                    "--fixture", os.path.join(HERE, "public_fixture.json"),
                    "--fixture-meta", os.path.join(HERE, "public_fixture_meta.json"),
                    "--expectations", os.path.join(HERE, "expectations.json"),
                    "--preference", os.path.join(HERE, "preference.json"),
                    "--output-dir", output,
                ])
            self.assertEqual(code, 0)
            with open(os.path.join(output, "trial_report.json"), encoding="utf-8") as handle:
                text = handle.read()
            self.assertNotIn(DECOY, text)
            saved = json.loads(text)
            self.assertEqual(saved["http"]["generation_http_requests"], 0)
            self.assertNotEqual(os.path.basename(output).lower()[:4], "sent")


class OneRequestTests(_NetworkGuard):
    def test_timeout_and_invalid_output_stop_after_one_call(self):
        jobs, meta, _expectations, _preference = load_public_jobs()
        expectations = frozen_expectations([job["id"] for job in jobs])
        preference = frozen_preference()
        calls = {"n": 0}

        def time_out(body, timeout):
            calls["n"] += 1
            adapter = runner.load_adapter(HERE)
            raise adapter.DeadlineError()

        report = runner.run_trial(
            HERE,
            jobs,
            sample=meta,
            expectations=expectations,
            preference=preference,
            live=True,
            transport=time_out,
        )
        self.assertEqual(calls["n"], 1)
        self.assertEqual(report["http"]["generation_http_requests"], 1)
        self.assertEqual(report["http"]["automatic_retries"], 0)
        self.assertEqual(report["http"]["stage"], "timeout")
        self.assertTrue(report["http"]["fallback"])
        self.assertEqual(report["structural"]["passed_items"], 0)
        self.assertEqual(report["structural"]["denominator_jobs"], 4)
        self.assertEqual(report["structural"]["rate"], 0)
        self.assertEqual(report["human_expectation"]["status"], "not_scored")
        self.assertFalse(report["trial_accepted"])

        calls["n"] = 0

        def bad_json(body, timeout):
            calls["n"] += 1
            self.assertEqual(body["model"], "deepseek-flash")
            self.assertIs(body["stream"], False)
            self.assertNotIn("thinking", body)
            self.assertNotIn("response_format", body)
            return completion("not json")

        failed = runner.run_trial(
            HERE,
            jobs,
            sample=meta,
            expectations=expectations,
            preference=preference,
            live=True,
            transport=bad_json,
        )
        self.assertEqual(calls["n"], 1)
        self.assertEqual(failed["http"]["stage"], "non_json")
        self.assertEqual(failed["http"]["generation_http_requests"], 1)
        self.assertEqual(failed["items"][0]["validation"], "non_json")
        self.assertIsNone(failed["items"][0]["output"])

    def test_second_transport_call_is_refused_before_the_inner_function(self):
        adapter = runner.load_adapter(HERE)
        calls = {"n": 0}

        def inner(body, timeout):
            calls["n"] += 1
            raise adapter.DeadlineError()

        guard = runner.OneShot(inner, adapter)
        with self.assertRaises(adapter.DeadlineError):
            guard({"model": "deepseek-flash", "stream": False}, 5)
        with self.assertRaises(adapter.ModelOutputError) as caught:
            guard({"model": "deepseek-flash", "stream": False}, 5)
        self.assertEqual(caught.exception.stage, "second_request_refused")
        self.assertEqual(calls["n"], 1)

    def test_other_model_never_calls_transport(self):
        jobs, meta, expectations, preference = load_public_jobs()

        def inner(body, timeout):
            raise AssertionError("transport called")

        report = runner.run_trial(
            HERE,
            jobs,
            sample=meta,
            expectations=frozen_expectations(["j1"]),
            preference=frozen_preference(),
            live=True,
            model="deepseek-chat",
            transport=inner,
        )
        self.assertEqual(report["http"]["generation_http_requests"], 0)
        self.assertEqual(report["http"]["stage"], "model_refused")
        self.assertEqual(expectations["status"], "not_frozen")

    def test_structural_pass_does_not_change_reason_schema_or_human_denominator(self):
        jobs, meta, _expectations, _preference = load_public_jobs()
        expectations = frozen_expectations(["j1", "j2"])
        calls = {"n": 0}

        def inner(body, timeout):
            calls["n"] += 1
            return completion(valid_content(jobs))

        report = runner.run_trial(
            HERE,
            jobs,
            sample=meta,
            expectations=expectations,
            preference=frozen_preference(),
            live=True,
            transport=inner,
        )
        self.assertEqual(calls["n"], 1)
        self.assertEqual(report["structural"]["passed_items"], 4)
        self.assertEqual(report["structural"]["denominator_jobs"], 4)
        self.assertEqual(report["structural"]["rate"], 1)
        self.assertEqual(report["human_expectation"]["status"], "scored")
        self.assertEqual(report["human_expectation"]["denominator"], 2)
        self.assertEqual(report["human_expectation"]["matches"], 1)
        self.assertEqual(report["human_expectation"]["missing_expectation_ids"], ["j3", "j4"])
        self.assertEqual(set(report["items"][0]["output"]), {"id", "group", "reason"})
        self.assertIsNone(report["schema"]["summary"])
        self.assertNotIn("summary", report["items"][0]["output"])
        self.assertFalse(report["trial_accepted"])
        self.assertEqual(report["cost"]["per_item_measured_cost"], "unknown")
        self.assertEqual(report["cost"]["batch_measured_cost"], "unknown")


class PricingTests(unittest.TestCase):
    def test_missing_cache_split_stays_unknown(self):
        report = pricing.cost_report(
            {"prompt_tokens": 100, "completion_tokens": 10},
            4,
            datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(report["per_item_measured_cost"], "unknown")
        self.assertEqual(report["batch_measured_cost"], "unknown")
        self.assertIsNone(report["amortized_estimate"])

    def test_off_peak_batch_is_computed_and_per_item_stays_unknown(self):
        moment = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(pricing.price_band(moment), "off_peak")
        report = pricing.cost_report(
            {
                "prompt_cache_hit_tokens": 1000000,
                "prompt_cache_miss_tokens": 0,
                "completion_tokens": 0,
            },
            4,
            moment,
        )
        self.assertEqual(report["per_item_measured_cost"], "unknown")
        self.assertEqual(report["batch_measured_cost"], "0.00300000")
        self.assertEqual(report["amortized_estimate"]["label"], "estimate_not_measurement")
        self.assertEqual(report["amortized_estimate"]["denominator_jobs"], 4)
        self.assertEqual(
            Decimal(report["amortized_estimate"]["amortized_per_item_usd"]),
            Decimal("0.00075000"),
        )

    def test_weekday_peak_clock_is_not_priced_as_fact(self):
        moment = datetime(2026, 11, 10, 2, 30, tzinfo=timezone.utc)
        self.assertEqual(pricing.price_band(moment), "unresolved_holiday")
        report = pricing.cost_report(
            {
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1000000,
                "completion_tokens": 0,
            },
            4,
            moment,
        )
        self.assertEqual(report["batch_measured_cost"], "unknown")
        self.assertEqual(report["amortized_estimate"]["label"], "estimate_not_measurement")
        self.assertEqual(report["amortized_estimate"]["off_peak"]["batch_usd"], "0.15000000")
        self.assertEqual(report["amortized_estimate"]["peak"]["batch_usd"], "0.30000000")


class AdapterReuseTests(unittest.TestCase):
    def test_bundle_adapter_keeps_official_endpoint_and_redirect_refusal(self):
        adapter = runner.load_adapter(HERE)
        self.assertEqual(adapter.API_URL, "https://api.deepseek.com/chat/completions")
        self.assertEqual(adapter.DEFAULT_MODEL, "deepseek-flash")
        self.assertIsNone(adapter._NoRedirect().redirect_request(
            None, None, 302, "found", {}, "https://evil.example/steal"
        ))
        jobs, _meta, _expectations, _preference = load_public_jobs()
        records, problem = adapter.normalize_records(jobs)
        self.assertEqual(problem, "")
        valid = json.loads(valid_content(jobs))
        self.assertEqual(len(adapter.validate_items(valid, records, "测试偏好")), 4)
        broken = json.loads(valid_content(jobs))
        broken["items"][0]["reason"] = "https://invented.invalid/job"
        with self.assertRaises(adapter.ModelOutputError) as caught:
            adapter.validate_items(broken, records, "测试偏好")
        self.assertEqual(caught.exception.stage, "generated_link")


if __name__ == "__main__":
    unittest.main()
