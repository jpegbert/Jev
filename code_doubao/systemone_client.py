# -*- coding: utf-8 -*-
"""
systemone_client.py
===================
通用 System One「决策模型」客户端，同一套代码兼容三种后端：

1. TypeSafe JEV 云 API      （设置环境变量 TYPESAFE_API_KEY）
2. 本地 Laya /v1/systemone   （设置 SYSTEMONE_BASE_URL=http://localhost:8000）
3. 离线 Mock                 （什么都不设置时自动启用，零依赖、可离线跑通全流程）

请求 / 响应遵循 JEV 官方 ``/v1/systemone`` schema：

    请求: {"state": <共享上下文>, "questions": {name: {type, instructions, criteria}}}
    响应: {"answers": {name: {type, ...概率/置信度字段}}, "_latency_ms": ...}

三个原语：
    - noul   : 是非判断，返回 0~1 的 yes 概率（0.5 表示“拿不准”）
    - choice : 从候选集（≤255）选一个，返回选中项 + 概率分布 + confidence
    - score  : 放到有序量表（2~10 级），返回概率加权分数（可落在两级之间）

依赖: 仅 Mock 模式零第三方依赖；真实 HTTP 模式需要 `pip install requests`。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from typing import Any, Dict, Optional

try:  # 真实 HTTP 才需要 requests
    import requests
except ImportError:  # pragma: no cover
    requests = None


DEFAULT_BASE_URL = os.getenv("SYSTEMONE_BASE_URL", "https://api.typesafe.ai")


# ============================================================================
# 真实 HTTP 客户端
# ============================================================================
class SystemOneClient:
    """POST {base_url}/v1/systemone，JEV 云与本地 Laya 通用。"""

    def __init__(self,
                 base_url: str = DEFAULT_BASE_URL,
                 api_key: Optional[str] = None,
                 timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("SYSTEMONE_API_KEY", "")
        self.timeout = timeout

    def ask(self,
            state: str,
            questions: Dict[str, Dict[str, Any]],
            **params) -> Dict[str, Any]:
        """对同一个 state 并行提出若干结构化问题，一次前向全部返回。"""
        if requests is None:
            raise RuntimeError(
                "真实 HTTP 调用需要 requests：`pip install requests`；"
                "或不配置任何 Key，自动走离线 Mock。")
        payload: Dict[str, Any] = {"state": state, "questions": questions}
        payload.update(params)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        t0 = time.time()
        resp = requests.post(f"{self.base_url}/v1/systemone",
                             json=payload, headers=headers,
                             timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        data["_latency_ms"] = round((time.time() - t0) * 1000, 1)
        return data

    def ask_many(self, state: str, *named_questions) -> Dict[str, Any]:
        """便捷写法：client.ask_many(state, ('q1', spec1), ('q2', spec2))。"""
        return self.ask(state, {name: spec for name, spec in named_questions})


# ============================================================================
# 离线 Mock 用到的文本工具
# ============================================================================
def _stable01(text: str) -> float:
    """把任意文本映射到 [0,1) 的稳定伪随机数（同输入同输出，便于复现）。"""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _cjk(s: str) -> str:
    return "".join(re.findall(r"[一-鿿0-9a-zA-Z]+", s or ""))


def _bigrams(s: str) -> set:
    t = _cjk(s)
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else set()


def _char_coverage(part: str, whole: str) -> float:
    """part 中有多少字符出现在 whole 里（按 part 归一，对短标签友好）。"""
    p = set(_cjk(part))
    if not p:
        return 0.0
    return len(p & set(_cjk(whole))) / len(p)


def _bigram_coverage(part: str, whole: str) -> float:
    bg = _bigrams(part)
    if not bg:
        return 0.0
    return len(bg & _bigrams(whole)) / len(bg)


def _char_overlap(a: str, b: str) -> float:
    sa, sb = set(a or ""), set(b or "")
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _slice_state(state: str, key: str) -> str:
    """批量场景：按问题名末尾数字，从共享 state 中切出对应的 ``[i] ...`` 行。"""
    m = re.search(r"(\d+)\D*$", key or "")
    if not m:
        return state
    mm = re.search(rf"^\s*\[[a-zA-Z_]*{m.group(1)}\]\s*(.+?)\s*$",
                   state or "", re.M)
    return mm.group(1) if mm else state


def _softmax_pick(scores: dict, temperature: float = 0.55,
                  uncertain: float = 0.30):
    """把 {选项: 原始分} 转成 (best, probs, confidence)。"""
    keys = list(scores.keys())
    mx = max(scores.values())
    exps = {k: math.exp((scores[k] - mx) / temperature) for k in keys}
    z = sum(exps.values())
    probs = {k: exps[k] / z for k in keys}
    best = max(probs, key=probs.get)
    # 原始最高分都很平庸 → 压置信度，倾向 other（若有）
    if mx <= uncertain and "other" in keys:
        best = "other"
    top = probs[best]
    if mx <= uncertain:
        top = min(top, 0.45)
    probs = {k: round(probs[k], 3) for k in keys}
    diff = round(1 - sum(probs.values()), 3)
    probs[best] = round(probs[best] + diff, 3)
    return best, probs, round(min(0.99, max(0.05, top)), 3)


# ============================================================================
# 离线 Mock 客户端（确定性的中文启发式，仅用于演示数据流 / 跑通结构）
# ============================================================================
class MockSystemOneClient(SystemOneClient):
    """用确定性启发式伪造结构化答案。

    它**不是模型**，只保证在没有 API Key / GPU 时，代码也能端到端跑通、
    看清 schema 与门控逻辑；接真实后端（JEV / Laya）后行为一致、结果更准。
    """

    CHOICE_HINTS = {
        "billing": ["账单", "发票", "退款", "扣款", "缴费", "支付", "到账",
                    "invoice", "refund", "payment"],
        "tech": ["故障", "报错", "崩溃", "502", "bug", "无法登录", "超时",
                 "宕机", "服务器", "数据库", "连接池", "technical", "outage"],
        "sales": ["购买", "套餐", "续费", "报价", "折扣", "价格", "升级会员",
                  "采购", "企业版", "咨询", "包含哪些", "sales", "purchase"],
        "hr": ["入职", "离职", "请假", "薪资", "招聘", "社保", "补交"],
        "spam": ["免费", "中奖", "点击领取", "限时", "广告", "红包", "稳赚",
                 "引流", "unsubscribe"],
    }

    # ---- Noul --------------------------------------------------------------
    def _noul(self, state: str, instr: str, key: str) -> Dict[str, Any]:
        text = _slice_state(state, key)

        # 1) 菜品判等
        if "同一道菜" in instr:
            m = re.search(
                r"菜\s*[ABabＡＢ][:：]\s*(?P<a>[^；;，,\n]+?)\s*[；;，,]\s*"
                r"菜\s*[ABabＡＢ][:：]\s*(?P<b>[^。.\n]+)", text)
            if m:
                a, b = self._normalize_dish(m.group("a")), \
                    self._normalize_dish(m.group("b"))
                rel = 0.5 * _char_coverage(a, b) + 0.5 * _bigram_coverage(a, b)
                p = 0.06 + 0.9 * rel
            else:  # 退化到整段字符重叠
                p = 0.3 + 0.4 * _stable01(text + instr)
            return {"type": "noul", "noul": round(min(0.98, max(0.03, p)), 3)}

        # 1b) 文章是否回答了标题对应的搜索意图（发布红绿灯）
        if "回答" in instr and "意图" in instr:
            t = re.search(r"标题[:：]\s*(.+)", text)
            b = re.search(r"正文[:：]\s*(.+)", text)
            if t and b:
                rel = _bigram_coverage(t.group(1), b.group(1))
                p = 0.15 + 0.8 * min(1.0, rel * 1.4)
            else:
                p = 0.4
            return {"type": "noul", "noul": round(min(0.97, p), 3)}

        # 2) 工具调用安全门
        if "安全" in instr or "授权" in instr:
            danger = ["生产数据库", "生产库", "drop production", "删库", "格式化",
                      "转账", "汇款", "清空生产", "封禁全部", "prod 数据库"]
            p = 0.08 if any(d in text for d in danger) else 0.9
            return {"type": "noul", "noul": round(p, 3)}

        # 3) 是否存在无来源断言（发布红绿灯）
        if "无来源" in instr or "断言" in instr:
            bad = ["毫无疑问", "无任何来源", "没有任何来源", "未标注", "毫无根据",
                   "没有内链", "未注明来源"]
            good = ["标注来源", "有据可查", "均来自", "引用了", "来源并标注", "注明来源"]
            p = 0.12 if any(g in text for g in good) and not \
                any(b in text for b in bad) else (0.88 if any(b in text for b in bad) else 0.4)
            return {"type": "noul", "noul": round(p, 3)}

        # 4) 历史片段是否仍需保留（上下文压缩）
        if "保留" in instr or ("有用" in instr) or ("仍" in instr and "需要" in instr):
            g = re.search(r"当前目标[:：]\s*(.+)", text)
            goal = g.group(1) if g else text
            line = text.split("\n")[-1] if re.search(r"[hH]\d+", key) else text
            distractor = ["天气", "无关", "三年前", "顺便", "闲聊"]
            useful = ["日志", "根因", "错误", "报错", "502", "db", "连接池", "配置",
                      "导致", "修复", "已确认", "打满", "超时", "支付"]
            goal_hits = ["502", "支付", "错误", "修复", "故障", "接口"]
            if any(d in line for d in distractor):
                p = 0.12
            elif _bigram_coverage(goal, line) >= 0.3:
                p = 0.9
            elif any(u in line for u in useful) and \
                    any(g in goal for g in goal_hits):
                p = 0.9
            else:
                p = 0.2
            return {"type": "noul", "noul": round(p, 3)}

        # 5) 销售话术是否匹配线索
        if "话术" in instr and "匹配" in instr:
            if ("学生" in text and any(w in text for w in ["合同", "SLA", "年度"])):
                p = 0.1
            elif ("采购" in text and any(w in text for w in ["私有化", "本地", "中文"])):
                p = 0.9
            elif ("贵" in text and any(w in text for w in ["降本", "省", "按调用量", "测算"])):
                p = 0.9
            else:
                p = 0.4 + 0.3 * _bigram_coverage(text.split("\n")[0], text)
            return {"type": "noul", "noul": round(min(0.95, p), 3)}

        # 6) 是否必须转人工
        if "转人工" in instr or "人工" in instr:
            angry = ["投诉", "宕机", "垃圾", "辱骂", "立刻", "两小时", "举报",
                     "再不", "严重", "法院", "起诉"]
            p = 0.9 if any(w in text for w in angry) else 0.2
            return {"type": "noul", "noul": round(p, 3)}

        # 7) 目标是否达成
        if "达成" in instr or "完成" in instr:
            done = ["已完成", "已经", "成功", "已删除", "已解决", "done", "finished"]
            p = 0.9 if any(w in text for w in done) else 0.35
            return {"type": "noul", "noul": round(p, 3)}

        # 8) 通用：退款/违规/危险/相关等关键词词典
        kw_map = {
            "退款": ["退款", "退钱", "refund", "重复扣款", "扣了两次", "退回", "到账"],
            "违规": ["违规", "违法", "色情", "赌博", "诈骗", "骚扰", "辱骂",
                    "垃圾广告", "稳赚", "红包", "spam", "scam", "illegal"],
            "相关": None,   # 走相关性
        }
        for k, kws in kw_map.items():
            if k in instr:
                if kws is None:
                    return {"type": "noul",
                            "noul": round(0.1 + 0.85 * _bigram_coverage(
                                text.split("\n")[0], text), 3)}
                hit = any(w.lower() in text.lower() for w in kws)
                p = 0.94 if hit else 0.10
                p += (_stable01(text + "|" + instr) - 0.5) * 0.04
                return {"type": "noul", "noul": round(min(0.99, max(0.01, p)), 3)}

        # 兜底
        p = 0.2 + 0.6 * _stable01(text + instr)
        return {"type": "noul", "noul": round(p, 3)}

    # ---- Choice ------------------------------------------------------------
    def _choice(self, state: str, q: Dict[str, Any], key: str) -> Dict[str, Any]:
        instr = q.get("instructions", "")
        text = _slice_state(state, key)
        criteria = q.get("criteria", {})
        if isinstance(criteria, list):
            criteria = {str(c): str(c) for c in criteria}

        # 特判：模型档位路由
        if "多强" in instr or "档位" in instr or ("模型" in instr and "选" in instr):
            return self._route_tier(text, criteria)

        # 特判：搜索意图
        if "搜索意图" in instr or ("意图" in instr and "informational" in str(criteria)):
            return self._search_intent(text, criteria)

        # 特判：浏览器下一步动作
        if "下一步" in instr and "元素" in instr:
            return self._browser_action(text, criteria)

        # 通用：候选标签/描述与文本的覆盖度
        scores = {}
        for k in criteria:
            if k == "other":
                scores[k] = 0.05
                continue
            desc = str(criteria.get(k, ""))
            s = 0.5 * _char_coverage(str(k), text) + 0.6 * _bigram_coverage(desc, text)
            s += 0.6 * _bigram_coverage(str(k), text)
            for canon, hints in self.CHOICE_HINTS.items():
                if canon in str(k).lower() or canon in desc.lower():
                    if any(h.lower() in text.lower() for h in hints):
                        s += 1.3
            s += 0.03 * _stable01(key + k)
            scores[k] = s
        best, probs, conf = _softmax_pick(scores)
        return {"type": "choice", "choice": best,
                "probabilities": probs, "confidence": conf}

    def _route_tier(self, text, criteria):
        hard = ["设计", "推导", "证明", "排查", "根因", "分布式", "死锁",
                "高并发", "方案", "架构", "对账", "限流", "复杂", "定位", "偶发"]
        easy = ["翻译", "错别字", "排序", "正面还是负面", "负面", "挑错", "改写",
                "格式", "一首", "短诗", "拼音", "哪个字母", "常识"]
        if any(w in text for w in hard):
            pick = "flagship"
        elif any(w in text for w in easy):
            pick = "cheap"
        else:
            pick = "balanced"
        pick = pick if pick in criteria else next(iter(criteria))
        return self._uniform_choice(pick, criteria, conf=0.82)

    def _search_intent(self, text, criteria):
        trans = ["购买", "下单", "下载", "注册", "优惠", "官网", "价格", "采购", "买"]
        comm = ["哪款", "哪个好", "推荐", "对比", "评测", "排行", "最好", "比较"]
        info = ["做法", "是什么", "什么是", "如何", "怎么", "了解", "概念", "教程"]
        if any(w in text for w in trans):
            pick = "transactional"
        elif any(w in text for w in comm):
            pick = "commercial"
        elif any(w in text for w in info):
            pick = "informational"
        else:
            pick = "informational"
        pick = pick if pick in criteria else next(iter(criteria))
        return self._uniform_choice(pick, criteria, conf=0.8)

    def _browser_action(self, text, criteria):
        if "London" in text and "日期" in text:
            pick = "click_search"
        elif "已填" in text:
            pick = "click_dest"
        else:
            pick = "click_origin"
        pick = pick if pick in criteria else next(iter(criteria))
        return self._uniform_choice(pick, criteria, conf=0.85)

    @staticmethod
    def _uniform_choice(pick, criteria, conf):
        keys = list(criteria.keys())
        rest = 1 - conf
        probs = {k: round(rest / max(1, len(keys) - 1), 3) for k in keys}
        probs[pick] = conf
        diff = round(1 - sum(probs.values()), 3)
        probs[pick] = round(probs[pick] + diff, 3)
        return {"type": "choice", "choice": pick,
                "probabilities": probs, "confidence": conf}

    # ---- Score -------------------------------------------------------------
    def _score(self, state: str, q: Dict[str, Any], key: str) -> Dict[str, Any]:
        instr = q.get("instructions", "")
        criteria = q.get("criteria", [])
        n_levels = max(2, min(10, len(criteria) if isinstance(
            criteria, (list, dict)) else 4))
        text = _slice_state(state, key)

        # 相关性：解析 query 与对应 [i] 文档，用 unigram+bigram 覆盖度
        if "相关" in instr:
            level = self._relevance_level(state, key, text)
        # 销售线索强度
        elif "成交意向" in instr or "线索" in instr or "强度" in instr:
            strong = ["采购", "本月", "负责人", "cto", "CTO", "抱怨", "太贵",
                      "方案", "要采购", "经理"]
            weak = ["学生", "了解概念", "随便看看", "只是"]
            if any(w in text for w in weak):
                level = 0.12
            elif any(w in text for w in strong):
                level = 0.85
            else:
                level = 0.5
        else:
            # 紧急/严重程度
            strong = ["紧急", "立刻", "马上", "严重", "崩溃", "生产", "critical",
                      "非常", "重复扣款", "投诉", "宕机", "两小时", "起诉"]
            weak = ["不急", "一般", "咨询", "了解", "顺便", "minor"]
            level = 0.5 + 0.3 * _stable01(text + instr)
            if any(w in text for w in strong):
                level += 0.45
            if any(w in text for w in weak):
                level -= 0.3
            level = min(1.0, max(0.0, level))

        score = min(1.0, max(0.0, level)) * (n_levels - 1)
        if score >= n_levels - 1:
            lo, hi = n_levels - 2, n_levels - 1
        else:
            lo, hi = int(score), int(score) + 1
        w_hi = score - lo
        dist = {str(lo): round(1 - w_hi, 2), str(hi): round(w_hi, 2)}
        conf = round(min(0.97, 0.6 + 0.35 * abs(level - 0.5) * 2), 3)
        return {"type": "score", "score": round(score, 2),
                "distribution": dist, "confidence": conf}

    @staticmethod
    def _normalize_dish(s: str) -> str:
        """把常见错别字 / 别名归一到同一写法（Mock 用的小词典，模拟同义知识）。"""
        aliases = [("宫爆", "宫保"), ("番茄", "西红柿"),
                   ("土豆片", "土豆丝"), ("盐煎肉", "回锅肉"),
                   ("麻辣豆腐", "麻婆豆腐")]
        out = s
        for a, b in aliases:
            out = out.replace(a, b)
        return out

    @staticmethod
    def _relevance_level(state, key, text):
        m = re.search(r"(\d+)\D*$", key or "")
        qm = re.search(r"查询[:：]\s*(.+?)\s*$", state or "", re.M)
        query = qm.group(1) if qm else state.split("\n")[0]
        doc = text
        if m:
            dm = re.search(rf"^\s*\[{m.group(1)}\]\s*(.+?)\s*$", state, re.M)
            if dm:
                doc = dm.group(1)
        cov = 0.5 * _char_coverage(query, doc) + 0.5 * _bigram_coverage(query, doc)
        return min(1.0, cov * 1.35)

    # ---- 入口 --------------------------------------------------------------
    def ask(self,
            state: str,
            questions: Dict[str, Dict[str, Any]],
            **params) -> Dict[str, Any]:
        t0 = time.time()
        answers: Dict[str, Any] = {}
        for name, q in questions.items():
            qtype = q.get("type", "noul")
            if qtype == "noul":
                answers[name] = self._noul(state, q.get("instructions", ""), name)
            elif qtype == "choice":
                answers[name] = self._choice(state, q, name)
            elif qtype == "score":
                answers[name] = self._score(state, q, name)
            else:
                answers[name] = {"type": qtype, "value": None}
        return {"answers": answers,
                "_latency_ms": round((time.time() - t0) * 1000, 2),
                "_mock": True}


# ============================================================================
# 工厂
# ============================================================================
def get_client(force_mock: bool = False) -> SystemOneClient:
    """按环境变量自动选择后端。"""
    if force_mock:
        return MockSystemOneClient()
    has_cloud = bool(os.getenv("TYPESAFE_API_KEY"))
    has_url = bool(os.getenv("SYSTEMONE_BASE_URL")) and \
        "api.typesafe.ai" not in os.getenv("SYSTEMONE_BASE_URL", "")
    if (has_cloud or has_url) and os.getenv("JEV_ALLOW_MOCK") != "1":
        return SystemOneClient()
    return MockSystemOneClient()


if __name__ == "__main__":
    c = get_client()
    demo = c.ask(
        state="用户说：我被重复扣款了，请立刻退款！",
        questions={
            "refund": {"type": "noul", "instructions": "用户是否要求退款？"},
            "department": {"type": "choice",
                           "instructions": "该分给哪个团队？",
                           "criteria": {"billing": "账单 退款 扣款",
                                        "tech": "技术 故障 bug",
                                        "sales": "销售 购买"}},
            "urgency": {"type": "score",
                        "instructions": "紧急程度？",
                        "criteria": ["不急", "一般", "紧急", "非常紧急"]},
        })
    print("backend =", c.__class__.__name__)
    print(json.dumps(demo, ensure_ascii=False, indent=2))
