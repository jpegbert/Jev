# -*- coding: utf-8 -*-
"""
14_ten_use_cases.py —— Jev 十个使用场景（综合自组内 PPT《Jev：10 个使用场景》）
==============================================================================

每个场景都是「选项已知、量大、要快」的有界判断；决策模型最有价值的不是快，
而是**明确告诉你哪些它没把握**，把不确定的那列交还人类，而不是悄悄猜错。

场景清单：
  01 收件箱自动分类          02 关键词意图归类       03 销售线索评分
  04 模型/工具路由(见13)     05 上下文压缩(见12)     06 浏览器操作选择
  07 搜索重排序(见09)        08 发布质量红绿灯       09 智能体护栏(见11)
  10 内容审核/工单分流

本文件把十个场景全部实现为可离线运行的函数（04/05/07/09 在专属文件里有更完整版）。
运行：python 14_ten_use_cases.py
"""
from __future__ import annotations

from systemone_client import get_client


# ---------------------------------------------------------------------------
# 01 收件箱自动分类（Choice；不确定留人工）
# ---------------------------------------------------------------------------
def uc01_inbox(client):
    folders = {"billing": "发票 账单 退款", "tech": "故障 bug 报错",
               "sales": "购买 套餐 报价", "hr": "入职 请假 招聘",
               "spam": "免费 中奖 广告", "other": "拿不准，转人工"}
    emails = ["您的发票已出具，请查收。", "限时免费领取会员，点击激活！",
              "生产库连接超时，订单 502。", "请周五前补交入职材料。"]
    qs = {f"m{i}": {"type": "choice",
                    "instructions": "这封邮件归入哪个文件夹？没把握选 other。",
                    "criteria": folders} for i in range(len(emails))}
    state = "\n".join(f"[m{i}] {e}" for i, e in enumerate(emails))
    ans = client.ask(state, qs)["answers"]
    print("[01 收件箱分类]")
    for i, e in enumerate(emails):
        a = ans[f"m{i}"]
        tag = "→ 人工" if a["choice"] == "other" else a["choice"]
        print(f"   {tag:<8} {e}")


# ---------------------------------------------------------------------------
# 02 关键词意图归类（Choice：informational / commercial / transactional）
# ---------------------------------------------------------------------------
def uc02_keyword_intent(client):
    intents = {"informational": "了解信息、教程、是什么、怎么做",
               "commercial": "对比、评测、哪个好、推荐",
               "transactional": "购买、下载、注册、下单、价格"}
    keywords = ["宫保鸡丁的做法", "蓝牙耳机哪款好", "iphone17 官网下单",
                "什么是决策模型", "云服务器优惠购买", "机械键盘 推荐 对比"]
    qs = {f"k{i}": {"type": "choice",
                    "instructions": "该搜索词属于哪种搜索意图？",
                    "criteria": intents} for i in range(len(keywords))}
    state = "\n".join(f"[k{i}] {k}" for i, k in enumerate(keywords))
    ans = client.ask(state, qs)["answers"]
    print("[02 关键词意图]")
    for i, k in enumerate(keywords):
        print(f"   {ans[f'k{i}']['choice']:<14} {k}")


# ---------------------------------------------------------------------------
# 03 销售线索评分（Score 强弱 + Noul 话术是否匹配；“不匹配”列最有价值）
# ---------------------------------------------------------------------------
def uc03_lead_scoring(client):
    leads = [
        ("某电商 CTO，明确说本月要采购决策模型方案", "强调可私有化部署与中文效果"),
        ("只是学生，留言说想了解概念", "推企业年度合同与 SLA"),
        ("中型搜索团队负责人，抱怨 LLM 判等太贵", "给出按调用量计费的降本测算"),
    ]
    print("[03 销售线索评分]")
    for person, pitch in leads:
        state = f"线索：{person}\n拟定话术：{pitch}"
        qs = {
            "quality": {"type": "score", "instructions": "该线索的成交意向强度？",
                        "criteria": ["弱", "中", "强"]},
            "fit": {"type": "noul",
                    "instructions": "拟定话术是否真的匹配这位线索的身份与诉求？"},
        }
        a = client.ask(state, qs)["answers"]
        mark = "❲话术不匹配，需重写❳" if a["fit"]["noul"] < 0.5 else "话术匹配"
        print(f"   强度={a['quality']['score']}  fit_p={a['fit']['noul']:<5} {mark}  {person[:14]}")


# ---------------------------------------------------------------------------
# 04 模型/工具路由（精简版，完整版见 13_model_router.py）
# ---------------------------------------------------------------------------
def uc04_router(client):
    prompts = ["把这句翻译成英文", "设计高并发交易系统的限流与对账方案"]
    q = {"type": "choice", "instructions": "任务需要多强的模型？",
         "criteria": {"cheap": "简单/常识/格式", "balanced": "常规问答/代码",
                      "flagship": "复杂推理/高风险"}}
    print("[04 模型路由]")
    for p in prompts:
        a = client.ask(p, {"t": q})["answers"]["t"]
        print(f"   -> {a['choice']:<9} {p}")


# ---------------------------------------------------------------------------
# 05 编码 Agent 上下文压缩（精简版，完整版见 12_context_compression.py）
# ---------------------------------------------------------------------------
def uc05_compression(client):
    goal = "修复支付 502"
    hist = ["支付服务日志：db 连接池打满导致 502", "顺便问下天气",
            "三年前无关部署记录", "连接池 max_connections=10 已打满"]
    state = f"当前目标：{goal}\n" + "\n".join(f"[h{i}] {h}" for i, h in enumerate(hist))
    qs = {f"h{i}": {"type": "noul",
                    "instructions": f"该历史对完成目标「{goal}」是否仍需保留？"}
          for i in range(len(hist))}
    ans = client.ask(state, qs)["answers"]
    kept = [hist[i] for i in range(len(hist)) if ans[f"h{i}"]["noul"] >= 0.7]
    print(f"[05 上下文压缩] 保留 {len(kept)}/{len(hist)} 条：{kept}")


# ---------------------------------------------------------------------------
# 06 浏览器操作选择（Choice：每步选“下一动作 + 元素”，观察后再要下一步）
# ---------------------------------------------------------------------------
def uc06_browser_action(client):
    print("[06 浏览器操作选择]（模拟 computer-use 循环）")
    steps = [
        "页面：Google Flights 首页；可见元素：[出发地输入框][目的地输入框][搜索按钮][登录链接]",
        "页面：已填 Zurich；可见元素：[目的地输入框][日期选择][搜索按钮]",
        "页面：已填 London、日期；可见元素：[搜索按钮][重置]",
    ]
    for obs in steps:
        q = {"type": "choice",
             "instructions": "为完成“订 Zurich→London 机票”，下一步应操作哪个元素？",
             "criteria": {"click_origin": "出发地输入框", "click_dest": "目的地输入框",
                          "click_search": "搜索按钮", "wait": "页面未就绪"}}
        a = client.ask(obs, {"act": q})["answers"]["act"]
        print(f"   观察→ {a['choice']}（执行后把新页面状态再交回决策模型）")


# ---------------------------------------------------------------------------
# 07 搜索结果重排序（精简版，完整版见 09_search_rerank.py）
# ---------------------------------------------------------------------------
def uc07_rerank(client):
    query = "宫保鸡丁做法"
    docs = ["宫保鸡丁家常做法与配料。", "今日股市行情。", "宫保鸡丁起源与口味。"]
    state = f"用户查询：{query}\n" + "\n".join(f"[{i}] {d}" for i, d in enumerate(docs))
    qs = {f"d{i}": {"type": "score", "instructions": "文档对查询的相关程度？",
                    "criteria": ["无关", "弱", "相关", "高度相关"]} for i in range(len(docs))}
    ans = client.ask(state, qs)["answers"]
    ranked = sorted(range(len(docs)), key=lambda i: ans[f"d{i}"]["score"], reverse=True)
    print("[07 搜索重排序]")
    for i in ranked:
        print(f"   score={ans[f'd{i}']['score']} {docs[i]}")


# ---------------------------------------------------------------------------
# 08 发布质量红绿灯（三个 Noul 并行 → 绿/黄/红分流）
# ---------------------------------------------------------------------------
def uc08_traffic_light(client):
    drafts = [
        ("如何做宫保鸡丁", "宫保鸡丁需要鸡胸肉、花生米、干辣椒……（含做法步骤）",
         "文中所有关键说法均来自食谱并标注来源，内链指向相关菜谱。"),
        ("2026 年决策模型盘点", "据说明年所有公司都会淘汰 LLM，这是毫无疑问的。",
         "该断言无任何来源，且没有内链。"),
    ]
    print("[08 发布红绿灯]")
    for title, body, meta in drafts:
        state = f"标题：{title}\n正文：{body}\n附注：{meta}"
        qs = {
            "answers_intent": {"type": "noul", "instructions": "文章是否真正回答了标题对应的搜索意图？"},
            "unsourced": {"type": "noul", "instructions": "文中是否存在无来源的断言？"},
            "links_ok": {"type": "noul", "instructions": "内链是否合理、相关？"},
        }
        a = client.ask(state, qs)["answers"]
        bad = a["unsourced"]["noul"] > 0.5
        good = (a["answers_intent"]["noul"] > 0.6 and a["links_ok"]["noul"] > 0.6 and not bad)
        light = "🟢 放行" if good else ("🔴 退回" if bad else "🟡 人工审")
        print(f"   {light}  intent={a['answers_intent']['noul']} "
              f"unsourced={a['unsourced']['noul']} links={a['links_ok']['noul']}  {title}")


# ---------------------------------------------------------------------------
# 09 智能体护栏（精简版，完整版见 11_agent_guardrail.py）
# ---------------------------------------------------------------------------
def uc09_guardrail(client):
    cases = ["删除测试环境临时文件", "删除生产数据库"]
    q = {"type": "noul", "instructions": "该操作是否安全且已被授权？"}
    print("[09 智能体护栏]")
    for c in cases:
        p = client.ask(f"请求：{c}", {"safe": q})["answers"]["safe"]["noul"]
        print(f"   {'✅放行' if p >= 0.8 else '🛑阻断'} safe_p={p}  {c}")


# ---------------------------------------------------------------------------
# 10 内容审核 / 工单分流（Choice 分类 + Score 紧急度 + Noul 是否转人工）
# ---------------------------------------------------------------------------
def uc10_ticket_moderation(client):
    items = ["加我微信领免费红包，稳赚不赔！！",
             "我的服务器已经宕机两小时，客户都在投诉，请立刻处理！",
             "想咨询一下企业版包含哪些功能。",
             "你这个垃圾软件，再不退款我就去投诉！"]
    cat = {"spam": "垃圾广告/引流", "abuse": "辱骂/骚扰",
           "billing": "退款/账单", "tech": "故障/宕机",
           "sales": "咨询/购买", "other": "其他"}
    print("[10 内容审核/工单分流]")
    for text in items:
        qs = {
            "category": {"type": "choice", "instructions": "这条内容/工单属于哪一类？",
                         "criteria": cat},
            "urgency": {"type": "score", "instructions": "紧急程度？",
                        "criteria": ["不急", "一般", "紧急", "非常紧急"]},
            "need_human": {"type": "noul", "instructions": "是否情绪激烈或高风险、必须立刻转人工？"},
        }
        a = client.ask(text, qs)["answers"]
        flag = "🔝置顶转人工" if a["need_human"]["noul"] >= 0.7 else "自动分流"
        print(f"   [{a['category']['choice']:<7}] 紧急={a['urgency']['score']} "
              f"{flag}  {text[:18]}")


def main():
    client = get_client()
    print(f"[backend] {client.__class__.__name__}\n")
    for fn in [uc01_inbox, uc02_keyword_intent, uc03_lead_scoring, uc04_router,
               uc05_compression, uc06_browser_action, uc07_rerank,
               uc08_traffic_light, uc09_guardrail, uc10_ticket_moderation]:
        fn(client)
        print()


if __name__ == "__main__":
    main()
