# 岗位筛选离线试验包

这个目录是一次小样本试验的最小运行包。它不包含仓库其余部分，也不包含密钥。

当前状态是离线准备。真实的 5 到 10 条岗位样本还没有到。`public_fixture.json` 里的 4 条是先前公开试点样本，不是那批真实 JD。`expectations.json` 和 `preference.json` 都是 `not_frozen`。在这两份文件被主管冻结之前，本包不会把任何输出当成验收通过，也不会发出生成请求。

## 这个包做什么

默认命令只读本地 JSON，写出 `trial_report.json`。生成 HTTP 请求数是 0。

模型输出如果将来出现，仍然只用既有 schema：`id`、`group`、`reason`。`reason` 是筛选短评，不是岗位摘要。摘要仍由仓库里的本地 `job_summary.py` 产生。本包不往 schema 里加摘要字段，也不放宽 `deepseek_screen.py` 的校验。

真实调用要同时满足下面四条，才会打到 `https://api.deepseek.com/chat/completions`：

1. 命令行显式带 `--live`。
2. `preference.json` 的 `status` 为 `frozen`，并且 `text` 非空。
3. `expectations.json` 的 `status` 为 `frozen`，并且每条都有合法的 `expected_group`。
4. 运行进程的环境变量 `DEEPSEEK_API_KEY` 已由那个环境自己注入。

即使四条都满足，整次试验也只有 1 次生成请求。自动重试是 0。超时、HTTP 错误或校验失败后立即停止，不会为了修输出再请求一次。模型只能是 `deepseek-flash`。重定向会被拒绝。

传输使用 `opener.open(request, timeout=超时秒数)`。`urllib` 的第二个位置参数是正文 data，不是超时。旧包把秒数当成 POST 正文，会在模型输出前抛出 `TypeError`。这份包已改掉。传输失败时结构通过率是 null，不是 0。只有服务真的返回了模型输出且校验失败，通过率才是 0。报告分开记录是否已尝试发送，以及服务是否返回了 HTTP 响应，并保留不含密钥的错误类型和短原因。

官方文档写明，省略 `thinking` 时 deepseek-flash 默认开启思考，默认努力为 high。思考和最终答案共用 `max_tokens`。2026-11-10 的 5 条真实请求 `finish_reason=length`，`completion_tokens` 正好是当时的 1400。这次请求显式发送 `thinking.type=disabled`。`reason` 最多 40 个字符。5 条的 `max_tokens` 是 528，20 条封顶 1024。这不是把 1400 调大。截断时报告保留最多 400 字的脱敏 content 摘录，以及思考文本的长度，不保存思考全文。

授权的两轴规则写在系统消息里。用户消息里的岗位正文和偏好仍是不可信数据，不能覆盖系统映射。正文不超过 8000 字时整段送出，并保留列表标签原串。超过上限则不发请求。报告里的 `outbound.messages` 是脱敏后的出站消息，带规则和用户内容的哈希，没有 Authorization。

本云端账号不能把 secret 转发进来。不要把密钥贴进聊天、写进文件，或从 Actions、仓库 secret、本机凭证里找。没有安全获准的执行环境时，就停在离线包。

## 离线运行

在本目录执行：

```bash
python3 runner.py \
  --fixture public_fixture.json \
  --fixture-meta public_fixture_meta.json \
  --expectations expectations.json \
  --preference preference.json \
  --output-dir ./out
```

`out/trial_report.json` 会记下每条原 id、必要原文、输出和校验原因。现在输出是空的，校验原因是 `not_called`。结构通过率的分母是输入岗位数，但状态是 `not_run`，比率是 null，不是 0%。人工期望一致率是 `not_scored`，分母是 null。逐条实测费用是 `unknown`。没有 usage 时，整批费用也是 `unknown`，不给均摊数字。

离线测试：

```bash
python3 test_offline.py
```

这组测试在进程里阻断 socket 连接。默认路径和未冻结的 `--live` 都不会发起连接。注入的假传输用来证明超时和坏 JSON 都只调用一次，第二次会被拒绝。

## 费用

单价来自 2026-11-10 读取的官方文档 https://api-docs.deepseek.com/quick_start/pricing/ 。deepseek-flash 每 100 万 token 的美元价格是：缓存命中输入，低峰 0.003，高峰 0.006。缓存未命中输入，低峰 0.15，高峰 0.3。输出，低峰 0.6，高峰 1.2。高峰时钟是周一到周五 UTC 01:00-04:00 和 06:00-10:00，官方不含中国法定节假日。本包没有节假日表，所以落在这个时钟里的工作日不定成实测价。

合批请求不能把 usage 拆到每一条岗位。逐条实测费用永远写 `unknown`。只有 usage 同时给出缓存命中、缓存未命中和输出 token，并且时段能确定时，才计算整批费用，并另给一行标明 `estimate_not_measurement` 的均摊。缺字段或时段不明时，整批实测费用也是 `unknown`。这不是节省比例，也没有把任务名称当成高耗证据。

## 以后的真实样本

样本由主管准备。样本可见授权确认之前，不要外采，也不要把这 4 条填进 `expectations.json`。冻结后的期望文件形如：

```json
{
  "status": "frozen",
  "frozen_at": "2026-11-10T00:00:00Z",
  "sample_id": "主管给出的样本编号",
  "items": [
    {"id": "原岗位 id", "expected_group": "优先"}
  ]
}
```

`expected_group` 只能是「优先」「观察」「不推荐」。它核对的是筛选分组，不是摘要句子。偏好文件要单独冻结 `text`。两份都冻结之后，才可以在已获准持有密钥的环境里加 `--live` 跑一次。

报告里的结构通过率分母是输入岗位数。校验是整批的，所以通过时是全部，失败时是 0，不会出现部分通过。人工一致率的分母只包括同时出现在样本和已冻结期望里的 id。缺期望的岗位另列，不进这个分母。`trial_accepted` 只有在结构通过且人工一致数等于该分母时才为 true。

## 尚未完成的生产接线

最终目标是小样本的真实筛选校验通过之后，再把 DeepSeek 设为岗位筛选的默认模型，合并旧适配，验证后切换生产。本包没有做这件事。默认关闭不算完成，只改岗位策略文字也不算完成。

最小接线还没做，门槛如下。

生产入口不在这个仓库的采集脚本里。`boss_cdp_raw.py` 不调用模型。Grok Bot 的岗位生产入口，以及 Mac 上 `/Users/fanjianyi/boss-zhipin-scraper` 可能未推送的脚本，都还没对照。在那份差异被核对之前，不能覆盖 Mac 上的本地代码、历史结果或 sent 标记。

以后获准接线时，只接这次这种高耗筛选调用。采集、通知、发送、固定催交、简单计数和轻量交互保持原样，不为它们新增 API 调用。批量摘要或简报要另一次授权。现有调用的 token 用量没有测过，不能靠任务名称宣称高耗或节省。

回滚门槛：生产入口保留原来的非 API 路径。切换提交可以单独撤回。撤回时不删除本地历史，不改 sent 标记。这个离线包本身撤回，只要不合并、不部署，默认行为就与现在一样。

## 本环境的阻塞

真实样本未到。人工期望未冻结。云端没有获准的密钥注入路径。因此这里不能执行 `--live`。下一步要等单独验收，不能自行跨步。
