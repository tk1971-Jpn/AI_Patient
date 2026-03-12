import json
import re
import sqlite3
import uuid
from datetime import datetime
from typing import Optional, List, Dict

import streamlit as st

DB_PATH = "sessions.sqlite"


# ---------------- DB ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        case_id TEXT,
        started_at TEXT,
        ended_at TEXT,
        model_name TEXT,
        prompt_version TEXT,
        temperature REAL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        t INTEGER,
        role TEXT,
        text TEXT,
        ts TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        session_id TEXT PRIMARY KEY,
        r_redflag INTEGER,
        r_structured INTEGER,
        r_diff INTEGER,
        r_tests INTEGER,
        r_initial_mgmt INTEGER,
        total INTEGER,
        confidence INTEGER,
        satisfaction INTEGER,
        summary TEXT,
        saved_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def log_session_start(session_id, case_id, model_name="rule_based_v1_1", prompt_version="v1.1", temperature=0.0):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions(session_id, case_id, started_at, ended_at, model_name, prompt_version, temperature) VALUES(?,?,?,?,?,?,?)",
        (session_id, case_id, datetime.now().isoformat(), None, model_name, prompt_version, temperature)
    )
    conn.commit()
    conn.close()


def log_session_end(session_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET ended_at=? WHERE session_id=?", (datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()


def log_turn(session_id, t, role, text):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO turns(session_id, t, role, text, ts) VALUES(?,?,?,?,?)",
        (session_id, t, role, text, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def upsert_evaluation(session_id, r1, r2, r3, r4, r5, conf, sat, summary):
    total = int(r1) + int(r2) + int(r3) + int(r4) + int(r5)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO evaluations(session_id, r_redflag, r_structured, r_diff, r_tests, r_initial_mgmt, total, confidence, satisfaction, summary, saved_at)
    VALUES(?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(session_id) DO UPDATE SET
      r_redflag=excluded.r_redflag,
      r_structured=excluded.r_structured,
      r_diff=excluded.r_diff,
      r_tests=excluded.r_tests,
      r_initial_mgmt=excluded.r_initial_mgmt,
      total=excluded.total,
      confidence=excluded.confidence,
      satisfaction=excluded.satisfaction,
      summary=excluded.summary,
      saved_at=excluded.saved_at
    """, (session_id, int(r1), int(r2), int(r3), int(r4), int(r5), total, int(conf), int(sat), summary, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return total


# ---------------- Case / Patient logic ----------------
def load_case(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """
    質問文の揺れを少し吸収するための簡易正規化
    """
    if not text:
        return ""

    text = text.lower()

    # 全角スペース → 半角
    text = text.replace("\u3000", " ")

    # 代表的な表記ゆれ補正
    replacements = {
        "おなか": "腹",
        "お腹": "腹",
        "みぎ": "右",
        "ひだり": "左",
        "きのう": "昨日",
        "いつ頃": "いつから",
        "いつごろ": "いつから",
        "発症時期": "いつから",
        "痛む場所": "どこ",
        "痛い場所": "どこ",
        "部位": "どこ",
        "吐き気": "嘔気",
        "むかつき": "嘔気",
        "もどした": "嘔吐",
        "吐いた": "嘔吐",
        "熱は": "発熱",
        "熱が": "発熱",
        "熱ありますか": "発熱",
        "ご飯": "食事",
        "食べたら": "食事",
        "食後": "食事後",
        "黄だん": "黄疸",
        "背中にひびく": "放散",
        "背中に響く": "放散",
        "しみるような": "性状",
        "差し込むような": "性状",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # 記号・句読点を少し整理
    text = re.sub(r"[、。,.，．!！?？「」『』（）()\[\]【】:：;；/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def contains_any(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords if kw)


def match_info_item(item: Dict, normalized_text: str) -> bool:
    """
    1つのinfo_bank itemにヒットするか判定
    使えるキー:
      - tags: 従来通りの部分一致
      - synonyms: tags の追加語彙
      - patterns: 全て含めばヒットする語群のリスト
        例: [["痛み", "どこ"], ["腹", "どこ"]]
    """
    tags = item.get("tags", [])
    synonyms = item.get("synonyms", [])
    patterns = item.get("patterns", [])

    if contains_any(normalized_text, tags):
        return True

    if contains_any(normalized_text, synonyms):
        return True

    for pattern_group in patterns:
        if all(term in normalized_text for term in pattern_group):
            return True

    return False


def match_info_bank(case: dict, user_text: str) -> List[str]:
    """
    info_bank からヒットした patient_answer を返す
    複数ヒット時は重複を除いて返す
    """
    normalized = normalize_text(user_text)
    hits = []

    for item in case.get("info_bank", []):
        if match_info_item(item, normalized):
            answer = item.get("patient_answer", "")
            if answer and answer not in hits:
                hits.append(answer)

    return hits


def asked_redflags(case: dict, user_text: str) -> bool:
    normalized = normalize_text(user_text)
    kws = case.get("safety_escalation", {}).get("redflag_keywords", [])
    extra_kws = [
        "息苦しい", "胸痛", "意識", "ふらつき", "冷や汗",
        "黄疸", "黒色便", "血便", "吐血", "激痛", "妊娠"
    ]
    return contains_any(normalized, kws + extra_kws)


def maybe_safety_escalation(case: dict, state: dict) -> Optional[str]:
    forced_turn = case.get("disclosure_rules", {}).get("forced_safety_escalation", {}).get("turn_if_no_redflag_check", 10)
    if state.get("safety_escalated"):
        return None
    if state.get("turn", 0) >= forced_turn and not state.get("redflag_checked", False):
        state["safety_escalated"] = True
        return case.get("safety_escalation", {}).get("message_if_triggered", None)
    return None


def is_question_about_test_results(user_text: str) -> bool:
    normalized = normalize_text(user_text)
    deny_words = [
        "採血結果", "検査結果", "CT結果", "エコー結果", "画像結果",
        "診断", "病名", "治療方針", "血液検査", "CT", "エコー", "US"
    ]
    return contains_any(normalized, deny_words)


def get_fallback_reply(user_text: str) -> str:
    normalized = normalize_text(user_text)

    if contains_any(normalized, ["どこ", "腹"]):
        return "痛いのはお腹です…。場所をもう少し詳しく聞いてもらえますか？"
    if contains_any(normalized, ["いつから", "昨日", "今日", "発症"]):
        return "時間のことですね…。もう少し聞き方を変えてもらえると答えやすいです。"
    if contains_any(normalized, ["痛み", "性状", "ズキズキ", "持続"]):
        return "痛みについてですね。どんな痛みか、続いているのかなどを聞いてもらえますか？"
    if contains_any(normalized, ["吐気", "嘔気", "嘔吐"]):
        return "気持ち悪さのことですね…。吐いたかどうかも含めて聞いてもらえますか？"

    return "すみません…その点はうまく答えられません。例えば、いつから・どこが・どんな痛みか、吐き気や熱があるか、みたいに聞いてもらえますか？"


def patient_reply(case: dict, user_text: str, state: dict) -> str:
    if is_question_about_test_results(user_text):
        return "それは自分では分からないので、先生に確認してもらえますか…？"

    if asked_redflags(case, user_text):
        state["redflag_checked"] = True

    hits = match_info_bank(case, user_text)
    if hits:
        return " ".join(hits)

    return get_fallback_reply(user_text)


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI模擬患者（腹痛：救急外来）", layout="wide")
init_db()

st.title("AI模擬患者（腹痛：救急外来）")
st.caption("自由問診 → オーダー → まとめ提出")

with st.sidebar:
    st.header("セッション")
    case_path = st.text_input("症例JSONパス", value="abd_pain_001.json")

    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    if st.button("新規セッション開始"):
        case = load_case(case_path)
        st.session_state.case = case
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.turn = 0
        st.session_state.messages = []
        st.session_state.state = {
            "turn": 0,
            "redflag_checked": False,
            "safety_escalated": False
        }
        st.session_state.revealed = {
            "exam": False,
            "labs": False,
            "us": False,
            "ct": False
        }
        st.session_state.summary_text = ""

        log_session_start(st.session_state.session_id, case["case_id"])

        opening = case["patient_profile"]["opening_statement"]
        st.session_state.messages.append(("patient", opening))
        log_turn(st.session_state.session_id, 0, "patient", opening)

        st.success(f"session_id: {st.session_state.session_id}")

    if st.button("セッション終了"):
        if st.session_state.session_id:
            log_session_end(st.session_state.session_id)
            st.info("セッションを終了しました。")
            st.session_state.session_id = None

if st.session_state.get("session_id") is None:
    st.warning("左の「新規セッション開始」を押してください。")
    st.stop()

case = st.session_state.case

col_left, col_right = st.columns([1.7, 1.0])

with col_left:
    st.subheader("医療面接")

    for role, text in st.session_state.messages:
        if role == "user":
            st.chat_message("user").write(text)
        elif role == "patient":
            st.chat_message("assistant").write(text)
        elif role == "result":
            st.chat_message("assistant").write(text)
        else:
            st.caption(text)

    user_text = st.chat_input("患者に質問してください（例：いつから？どこが？吐き気は？黄疸は？）")
    if user_text:
        st.session_state.turn += 1
        t = st.session_state.turn

        st.session_state.messages.append(("user", user_text))
        log_turn(st.session_state.session_id, t, "user", user_text)

        st.session_state.state["turn"] = t
        reply = patient_reply(case, user_text, st.session_state.state)
        st.session_state.messages.append(("patient", reply))
        log_turn(st.session_state.session_id, t, "patient", reply)

        esc = maybe_safety_escalation(case, st.session_state.state)
        if esc:
            st.session_state.turn += 1
            t2 = st.session_state.turn
            st.session_state.messages.append(("patient", esc))
            log_turn(st.session_state.session_id, t2, "patient", esc)

        st.rerun()

    st.divider()
    st.subheader("オーダー")
    st.caption("※患者は検査結果を知らない設定です。ここは医療者としてオーダーした結果を開示します。")

    order_col1, order_col2 = st.columns(2)

    with order_col1:
        if st.button("身体診察（腹部診察）", use_container_width=True):
            st.session_state.turn += 1
            t = st.session_state.turn
            st.session_state.revealed["exam"] = True
            log_turn(st.session_state.session_id, t, "order", "身体診察（腹部診察）")
            result = case["exam_findings"]["abd_exam"]
            st.session_state.messages.append(("result", f"【身体診察】{result}"))
            log_turn(st.session_state.session_id, t, "result", f"身体診察: {result}")
            st.rerun()

        if st.button("採血（基本＋肝胆膵）", use_container_width=True):
            st.session_state.turn += 1
            t = st.session_state.turn
            st.session_state.revealed["labs"] = True
            log_turn(st.session_state.session_id, t, "order", "採血（CBC/CRP/生化/LFT/Amylase/妊娠）")
            labs = case["orders"]["labs"]["results"]
            result = " / ".join([f"{k}: {v}" for k, v in labs.items()])
            st.session_state.messages.append(("result", f"【採血】{result}"))
            log_turn(st.session_state.session_id, t, "result", f"採血: {result}")
            st.rerun()

    with order_col2:
        if st.button("腹部エコー（US）", use_container_width=True):
            st.session_state.turn += 1
            t = st.session_state.turn
            st.session_state.revealed["us"] = True
            log_turn(st.session_state.session_id, t, "order", "腹部エコー（US）")
            result = case["orders"]["imaging"]["us_abd"]
            st.session_state.messages.append(("result", f"【US】{result}"))
            log_turn(st.session_state.session_id, t, "result", f"US: {result}")
            st.rerun()

        if st.button("腹部CT", use_container_width=True):
            st.session_state.turn += 1
            t = st.session_state.turn
            st.session_state.revealed["ct"] = True
            log_turn(st.session_state.session_id, t, "order", "腹部CT")
            result = case["orders"]["imaging"]["ct_abd"]
            st.session_state.messages.append(("result", f"【CT】{result}"))
            log_turn(st.session_state.session_id, t, "result", f"CT: {result}")
            st.rerun()

    st.divider()
    st.subheader("まとめ提出")
    st.caption("最後にこの症例のまとめを提出してください。提出内容はSQLiteに保存されます。")

    st.session_state.summary_text = st.text_area(
        "まとめ（鑑別3つ、検査、初期対応、患者説明）",
        value=st.session_state.summary_text,
        height=180
    )

    if st.button("まとめを提出", use_container_width=True):
        st.session_state.turn += 1
        t = st.session_state.turn
        summary_value = st.session_state.summary_text if st.session_state.summary_text else "(empty)"
        log_turn(st.session_state.session_id, t, "summary", summary_value)
        st.success("まとめを保存しました。")

with col_right:
    st.subheader("患者情報")

    p = case["patient_profile"]
    s = case["initial_state"]

    st.markdown("### 基本情報")
    st.write("**設定**: 救急外来")
    st.write(f"**症例ID**: {case.get('case_id', '—')}")
    st.write(f"**年齢**: {p.get('age', '—')}")
    st.write(f"**性別**: {p.get('sex', '—')}")
    st.write(f"**主訴**: {p.get('chief_complaint', '—')}")

    st.divider()

    st.markdown("### 来院時情報")
    st.write("**来院時バイタル**")
    st.json(s["vitals"])
    st.write(f"**外観**: {s.get('appearance', '—')}")
    st.write(f"**トリアージメモ**: {s.get('triage_note', '—')}")

    st.divider()

    st.markdown("### 注意")
    st.info("この欄は、学生に最初から提示してよい基本情報のみを表示しています。")