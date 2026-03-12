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

    # ルーブリック/自己評価/まとめを保存
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

    # 何も引っかからない時は問診を促す（教育的）
    return "すみません…その点はよく分からなくて。例えば、いつから・どこが・どんな痛みか、みたいに聞いてもらえますか？"

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI模擬患者（腹痛：救急外来）", layout="wide")
init_db()

st.title("AI模擬患者（腹痛：救急外来）— 自由問診 → まとめ提出（研究ログ保存）")

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
        st.session_state.state = {"turn": 0, "redflag_checked": False, "safety_escalated": False}
        st.session_state.revealed = {"exam": False, "labs": False, "us": False, "ct": False}

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

# ---- Layout ----
col1, col2, col3 = st.columns([1.1, 1.7, 1.2])

with col1:
    st.subheader("患者カード")
    p = case["patient_profile"]
    s = case["initial_state"]
    st.write(f"**設定**: 救急外来")
    st.write(f"**年齢/性別**: {p['age']} / {p['sex']}")
    st.write(f"**主訴**: {p['chief_complaint']}")
    st.write("**来院時バイタル**")
    st.json(s["vitals"])
    st.write(f"**外観**: {s['appearance']}")
    st.write(f"**トリアージメモ**: {s['triage_note']}")

    st.divider()
    st.subheader("オーダー（教員/学生操作）")
    st.caption("※患者は検査結果を知らない設定。ここは“医療者としてオーダーした”扱いで結果が開示されます。")

    # 身体診察
    if st.button("身体診察（腹部診察）"):
        st.session_state.turn += 1
        t = st.session_state.turn
        st.session_state.revealed["exam"] = True
        log_turn(st.session_state.session_id, t, "order", "身体診察（腹部診察）")
        result = case["exam_findings"]["abd_exam"]
        st.session_state.messages.append(("result", f"【身体診察】{result}"))
        log_turn(st.session_state.session_id, t, "result", f"身体診察: {result}")

    # 採血
    if st.button("採血（基本＋肝胆膵）"):
        st.session_state.turn += 1
        t = st.session_state.turn
        st.session_state.revealed["labs"] = True
        log_turn(st.session_state.session_id, t, "order", "採血（CBC/CRP/生化/LFT/Amylase/妊娠）")
        labs = case["orders"]["labs"]["results"]
        result = " / ".join([f"{k}: {v}" for k, v in labs.items()])
        st.session_state.messages.append(("result", f"【採血】{result}"))
        log_turn(st.session_state.session_id, t, "result", f"採血: {result}")

    # US
    if st.button("腹部エコー（US）"):
        st.session_state.turn += 1
        t = st.session_state.turn
        st.session_state.revealed["us"] = True
        log_turn(st.session_state.session_id, t, "order", "腹部エコー（US）")
        result = case["orders"]["imaging"]["us_abd"]
        st.session_state.messages.append(("result", f"【US】{result}"))
        log_turn(st.session_state.session_id, t, "result", f"US: {result}")

    # CT
    if st.button("腹部CT"):
        st.session_state.turn += 1
        t = st.session_state.turn
        st.session_state.revealed["ct"] = True
        log_turn(st.session_state.session_id, t, "order", "腹部CT")
        result = case["orders"]["imaging"]["ct_abd"]
        st.session_state.messages.append(("result", f"【CT】{result}"))
        log_turn(st.session_state.session_id, t, "result", f"CT: {result}")

with col2:
    st.subheader("問診チャット（自由問診）")

    # 表示
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

        # patient reply
        st.session_state.state["turn"] = t
        reply = patient_reply(case, user_text, st.session_state.state)
        st.session_state.messages.append(("patient", reply))
        log_turn(st.session_state.session_id, t, "patient", reply)

        # safety escalation (if needed)
        esc = maybe_safety_escalation(case, st.session_state.state)
        if esc:
            st.session_state.turn += 1
            t2 = st.session_state.turn
            st.session_state.messages.append(("patient", esc))
            log_turn(st.session_state.session_id, t2, "patient", esc)

        st.rerun()

    st.divider()
    st.subheader("まとめ提出（研究の主アウトカム）")
    st.caption("最後にこの症例のまとめを提出してください。提出内容はSQLiteに保存されます。")
    summary_text = st.text_area("まとめ（鑑別3つ、検査、初期対応、患者説明）", height=180)

    if st.button("まとめを提出"):
        st.session_state.turn += 1
        t = st.session_state.turn
        log_turn(st.session_state.session_id, t, "summary", summary_text if summary_text else "(empty)")
        st.success("まとめを保存しました。右の評価も一緒に「評価を保存」を押すと研究データが完成します。")

with col3:
    st.subheader("評価（教員/自己評価）")
    st.caption("0–2点×5項目（合計10点）。必要なら後で5段階化できます。")

    r1 = st.slider("赤旗確認", 0, 2, 0)
    r2 = st.slider("病歴の構造化（OPQRST等）", 0, 2, 0)
    r3 = st.slider("鑑別と優先順位", 0, 2, 0)
    r4 = st.slider("検査プラン", 0, 2, 0)
    r5 = st.slider("初期対応＋説明", 0, 2, 0)

    total_preview = r1 + r2 + r3 + r4 + r5
    st.metric("合計（プレビュー）", total_preview)

    st.divider()
    conf = st.slider("自己評価：自信（0-10）", 0, 10, 5)
    sat = st.slider("自己評価：満足度（0-10）", 0, 10, 5)

    st.divider()
    st.subheader("保存")
    st.caption("評価＋自己評価＋まとめをSQLiteに保存します（研究用データ完成）。")

    # まとめはcol2の入力を再利用できないので、簡易に直近summaryをturnsから読むのは重い。
    # ここではユーザーに「まとめ提出→評価保存」を推奨し、summary_textが空なら空で保存します。
    # （運用上は問題なし。必要なら state に summary_text を保存する形に変更可能）
    if "last_summary" not in st.session_state:
        st.session_state.last_summary = ""

    # col2のsummary_textはスコープ外なので、ここで別欄も用意（安全）
    st.session_state.last_summary = st.text_area("（保存用）まとめテキスト", value=st.session_state.last_summary, height=120)

    if st.button("評価を保存"):
        total = upsert_evaluation(
            st.session_state.session_id,
            r1, r2, r3, r4, r5,
            conf, sat,
            st.session_state.last_summary
        )
        st.success(f"評価を保存しました（合計 {total} / 10）。")
        st.info("この時点で「ログ（turns）」＋「評価（evaluations）」が揃い、pre/post研究が可能です。")
