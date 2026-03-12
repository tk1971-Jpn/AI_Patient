import json
import sqlite3
import uuid
from datetime import datetime
import streamlit as st
from typing import Optional

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

    # ルーブリック/自己評価/まとめを保存（将来用に残す）
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


def log_session_start(session_id, case_id, model_name="TBD", prompt_version="v0", temperature=0.2):
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


# ---------------- Case / Patient logic (rule-based MVP) ----------------
def load_case(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_info_bank(case: dict, user_text: str):
    """tags に含まれる語が user_text に出たら、その患者回答を返す（複数ヒットは連結）"""
    hits = []
    for item in case.get("info_bank", []):
        for tag in item.get("tags", []):
            if tag and (tag in user_text):
                hits.append(item["patient_answer"])
                break
    return hits


def asked_redflags(case: dict, user_text: str) -> bool:
    kws = case.get("safety_escalation", {}).get("redflag_keywords", [])
    return any(kw in user_text for kw in kws)


def maybe_safety_escalation(case: dict, state: dict) -> Optional[str]:
    """赤旗確認が一定ターンまで無い場合に1回だけ訴える"""
    forced_turn = case.get("disclosure_rules", {}).get("forced_safety_escalation", {}).get("turn_if_no_redflag_check", 10)
    if state.get("safety_escalated"):
        return None
    if state.get("turn", 0) >= forced_turn and not state.get("redflag_checked", False):
        state["safety_escalated"] = True
        return case.get("safety_escalation", {}).get("message_if_triggered", None)
    return None


def patient_reply(case: dict, user_text: str, state: dict) -> str:
    # ルール：検査値・画像は患者は知らない
    deny_words = ["採血結果", "検査結果", "CT結果", "エコー結果", "診断", "病名", "治療方針"]
    if any(w in user_text for w in deny_words):
        return "それは自分では分からないので、先生に確認してもらえますか…？"

    # 赤旗確認があったかカウント
    if asked_redflags(case, user_text):
        state["redflag_checked"] = True

    hits = match_info_bank(case, user_text)
    if hits:
        return " ".join(hits)

    # 何も引っかからない時は問診を促す
    return "すみません…その点はよく分からなくて。例えば、いつから・どこが・どんな痛みか、みたいに聞いてもらえますか？"


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

        # オープニング
        opening = case["patient_profile"]["opening_statement"]
        st.session_state.messages.append(("patient", opening))
        log_turn(st.session_state.session_id, 0, "patient", opening)

        st.success(f"session_id: {st.session_state.session_id}")

    if st.button("セッション終了"):
        if st.session_state.session_id:
            log_session_end(st.session_state.session_id)
            st.info("セッションを終了しました。")
            st.session_state.session_id = None

# ガード
if st.session_state.get("session_id") is None:
    st.warning("左の「新規セッション開始」を押してください。")
    st.stop()

case = st.session_state.case

# ---- Layout: 2 columns ----
col_left, col_right = st.columns([1.7, 1.0])

# ---------------- Left: Interview ----------------
with col_left:
    st.subheader("医療面接")

    # チャット表示
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

# ---------------- Right: Patient info ----------------
with col_right:
    st.subheader("患者情報")

    p = case["patient_profile"]
    s = case["initial_state"]

    st.markdown("### 基本情報")
    st.write(f"**設定**: 救急外来")
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