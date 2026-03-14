import csv
import io
import json
import os
import re
import sqlite3
import uuid
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========================================================
# 設定
# =========================================================
DB_PATH = "sessions.sqlite"
APP_TITLE = "AI模擬患者 - 救急外来問診トレーニング"
DEFAULT_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-mini")
PROMPT_VERSION = "app7_hybrid_research_v1"
TEMPERATURE = 0.2

# ---------------------------------------------------------
# 症例設定（急性胆嚢炎）
# ---------------------------------------------------------
CASE_DATA = {
    "case_id": "acute_cholecystitis_001",
    "title": "救急外来：急性胆嚢炎",
    "difficulty": "medical_student",
    "patient": {
        "name": "田中 恒一",
        "age": 52,
        "sex": "男性",
        "occupation": "会社員",
        "marital_status": "既婚",
    },
    "chief_complaint": "右上腹部痛と発熱です。",
    "history_of_present_illness": {
        "onset": "昨日の夕方ごろからです。",
        "course": "最初はなんとなくみぞおちのあたりが重い感じでしたが、だんだん右上腹部が強く痛くなってきました。",
        "location": "右上腹部です。",
        "radiation": "背中にも少しひびく感じがあります。",
        "severity": "じっとしていてもつらいくらいで、10段階なら7くらいです。",
        "fever": "熱っぽくて、家で測ったら38度くらいありました。",
        "nausea": "少し気持ち悪さがあります。",
        "vomiting": "吐いてはいません。",
        "appetite": "食欲はあまりありません。",
        "trigger": "脂っこいものを食べたあとに悪くなった気がします。",
        "relieving_factors": "楽になる姿勢はあまりありません。",
        "aggravating_factors": "体を動かすと少し響きます。",
        "bowel": "便通は普段どおりです。",
        "urination": "尿は普通です。",
        "similar_episode": "ここまで強いのは初めてですが、以前にも脂っこいもののあとに右上腹部が少し痛んだことはありました。",
    },
    "past_history": {
        "medical": "高血圧があります。",
        "surgery": "手術は受けたことがありません。",
        "admission": "入院歴はありません。",
    },
    "medications": "血圧の薬を1種類飲んでいます。詳しい名前はお薬手帳を見ればわかります。",
    "allergies": "薬のアレルギーはありません。",
    "family_history": "特に大きな病気は聞いていません。",
    "social_history": {
        "smoking": "たばこは吸いません。",
        "alcohol": "お酒は週に2〜3回、ビールを1〜2本くらいです。",
        "living": "妻と2人暮らしです。",
    },
    "review_of_systems": {
        "chest_pain": "胸の痛みはありません。",
        "dyspnea": "息苦しさはありません。",
        "headache": "頭痛はありません。",
        "diarrhea": "下痢はありません。",
        "jaundice": "自分では黄疸はわかりません。",
    },
    "emotion": {
        "concern": "胆石か何かでしょうか。かなり痛いので心配です。",
        "expectation": "原因をはっきりさせて、痛みを取ってほしいです。",
    },
}

# ---------------------------------------------------------
# 辞書 / ルール
# ---------------------------------------------------------
INTENT_RULES = {
    "chief_complaint": {
        "patterns": [
            r"本日はどうされましたか",
            r"今日はどうされましたか",
            r"どうしましたか",
            r"どのような症状",
            r"主訴",
            r"いちばん困っていること",
            r"今日はどうされ",
            r"どうされました",
        ],
        "answer": CASE_DATA["chief_complaint"],
    },
    "onset": {
        "patterns": [r"いつから", r"発症", r"何時から", r"始まったのはいつ", r"症状はいつから"],
        "answer": CASE_DATA["history_of_present_illness"]["onset"],
    },
    "course": {
        "patterns": [r"どんな経過", r"経過", r"だんだん", r"悪くな", r"よくな", r"その後"],
        "answer": CASE_DATA["history_of_present_illness"]["course"],
    },
    "location": {
        "patterns": [r"どこが痛", r"痛む場所", r"場所は", r"部位", r"どのあたり"],
        "answer": CASE_DATA["history_of_present_illness"]["location"],
    },
    "radiation": {
        "patterns": [r"ひびく", r"放散", r"広が", r"背中", r"肩にも"],
        "answer": CASE_DATA["history_of_present_illness"]["radiation"],
    },
    "severity": {
        "patterns": [r"痛みの強さ", r"どれくらい痛", r"何点", r"10段階", r"痛みは強い"],
        "answer": CASE_DATA["history_of_present_illness"]["severity"],
    },
    "fever": {
        "patterns": [r"熱", r"発熱", r"何度", r"体温"],
        "answer": CASE_DATA["history_of_present_illness"]["fever"],
    },
    "nausea": {
        "patterns": [r"吐き気", r"気持ち悪", r"むかむか"],
        "answer": CASE_DATA["history_of_present_illness"]["nausea"],
    },
    "vomiting": {
        "patterns": [r"吐い", r"嘔吐"],
        "answer": CASE_DATA["history_of_present_illness"]["vomiting"],
    },
    "appetite": {
        "patterns": [r"食欲", r"食べられ", r"ご飯は"],
        "answer": CASE_DATA["history_of_present_illness"]["appetite"],
    },
    "trigger": {
        "patterns": [r"きっかけ", r"原因", r"何か思い当たる", r"誘因"],
        "answer": CASE_DATA["history_of_present_illness"]["trigger"],
    },
    "relieving_factors": {
        "patterns": [r"楽になる", r"よくなる", r"軽くなる", r"改善する", r"和らぐ"],
        "answer": CASE_DATA["history_of_present_illness"]["relieving_factors"],
    },
    "aggravating_factors": {
        "patterns": [r"悪化", r"ひどくなる", r"増悪", r"動くと", r"痛みが強くなる"],
        "answer": CASE_DATA["history_of_present_illness"]["aggravating_factors"],
    },
    "bowel": {
        "patterns": [r"便", r"便通", r"下痢", r"便秘"],
        "answer": CASE_DATA["history_of_present_illness"]["bowel"],
    },
    "urination": {
        "patterns": [r"尿", r"排尿", r"おしっこ"],
        "answer": CASE_DATA["history_of_present_illness"]["urination"],
    },
    "similar_episode": {
        "patterns": [r"今回が初めて", r"同じような", r"以前にも", r"似たような", r"繰り返し"],
        "answer": CASE_DATA["history_of_present_illness"]["similar_episode"],
    },
    "past_history": {
        "patterns": [
            r"既往歴",
            r"これまで何か病気",
            r"今まで何か病気",
            r"持病",
            r"病気をされたこと",
            r"大きな病気",
            r"治療中の病気",
        ],
        "answer": CASE_DATA["past_history"]["medical"],
    },
    "surgery": {
        "patterns": [r"手術歴", r"手術したこと", r"今まで手術", r"オペしたこと"],
        "answer": CASE_DATA["past_history"]["surgery"],
    },
    "admission": {
        "patterns": [r"入院歴", r"入院したこと"],
        "answer": CASE_DATA["past_history"]["admission"],
    },
    "medications": {
        "patterns": [r"内服", r"飲んでいる薬", r"お薬", r"薬は", r"常用薬"],
        "answer": CASE_DATA["medications"],
    },
    "allergies": {
        "patterns": [r"アレルギー", r"薬で具合が悪く", r"薬疹", r"食べ物のアレルギー"],
        "answer": CASE_DATA["allergies"],
    },
    "family_history": {
        "patterns": [r"家族歴", r"ご家族で", r"家族に", r"血縁"],
        "answer": CASE_DATA["family_history"],
    },
    "smoking": {
        "patterns": [r"たばこ", r"喫煙", r"吸っていますか", r"吸いますか"],
        "answer": CASE_DATA["social_history"]["smoking"],
    },
    "alcohol": {
        "patterns": [r"お酒", r"飲酒", r"アルコール"],
        "answer": CASE_DATA["social_history"]["alcohol"],
    },
    "living": {
        "patterns": [r"誰と住", r"同居", r"一人暮らし", r"家族構成", r"生活状況"],
        "answer": CASE_DATA["social_history"]["living"],
    },
    "chest_pain": {
        "patterns": [r"胸の痛み", r"胸痛"],
        "answer": CASE_DATA["review_of_systems"]["chest_pain"],
    },
    "dyspnea": {
        "patterns": [r"息苦し", r"呼吸苦", r"息切れ"],
        "answer": CASE_DATA["review_of_systems"]["dyspnea"],
    },
    "headache": {
        "patterns": [r"頭痛", r"頭は痛"],
        "answer": CASE_DATA["review_of_systems"]["headache"],
    },
    "diarrhea": {
        "patterns": [r"下痢", r"軟便"],
        "answer": CASE_DATA["review_of_systems"]["diarrhea"],
    },
    "jaundice": {
        "patterns": [r"黄疸", r"皮膚が黄色", r"目が黄色"],
        "answer": CASE_DATA["review_of_systems"]["jaundice"],
    },
    "concern": {
        "patterns": [r"心配", r"不安", r"気がかり"],
        "answer": CASE_DATA["emotion"]["concern"],
    },
    "expectation": {
        "patterns": [r"どうしてほしい", r"希望", r"期待", r"何を望", r"してほしい"],
        "answer": CASE_DATA["emotion"]["expectation"],
    },
}

NORMALIZATION_MAP = {
    "どうされましたか": "本日はどうされましたか",
    "どうしましたか": "本日はどうされましたか",
    "今日はどうされましたか": "本日はどうされましたか",
    "病気をされたこと": "既往歴",
    "これまで何か病気": "既往歴",
    "今まで何か病気": "既往歴",
    "飲んでる薬": "飲んでいる薬",
    "タバコ": "たばこ",
    "おなか": "腹部",
    "みぎうえ": "右上",
}

SYSTEM_PROMPT = f"""
あなたは医療面接訓練用アプリに組み込まれた模擬患者です。
役割は「患者として自然に受け答えすること」です。

絶対ルール:
- 症例設定にない新事実を勝手に作らない。
- わからない場合は、患者らしく「わかりません」「特にありません」など簡潔に答える。
- 回答は1〜3文で短く、日本語で自然に。
- 医師のような説明や診断名の断定はしない。
- 質問に対応する症例情報が明確にある場合のみ、その範囲で答える。
- 患者の口調で答える。

このアプリの prompt version は {PROMPT_VERSION} です。
""".strip()

# =========================================================
# DB
# =========================================================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with closing(get_conn()) as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                case_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                model_name TEXT,
                prompt_version TEXT,
                temperature REAL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                turn_index INTEGER,
                role TEXT,
                content TEXT,
                source TEXT,
                intent TEXT,
                created_at TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fallback_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                turn_index INTEGER,
                question TEXT,
                normalized_question TEXT,
                ai_answer TEXT,
                matched_intent TEXT,
                created_at TEXT
            )
            """
        )

        conn.commit()


def create_session(model_name: str) -> str:
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO sessions (session_id, case_id, started_at, ended_at, model_name, prompt_version, temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, CASE_DATA["case_id"], now, None, model_name, PROMPT_VERSION, TEMPERATURE),
        )
        conn.commit()
    return session_id


def close_session(session_id: str):
    now = datetime.now().isoformat(timespec="seconds")
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE sessions SET ended_at = ? WHERE session_id = ?", (now, session_id))
        conn.commit()


def save_turn(session_id: str, turn_index: int, role: str, content: str, source: str, intent: str = ""):
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO turns (session_id, turn_index, role, content, source, intent, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, turn_index, role, content, source, intent, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()


def save_fallback_event(session_id: str, turn_index: int, question: str, normalized_question: str, ai_answer: str, matched_intent: str = ""):
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO fallback_events (session_id, turn_index, question, normalized_question, ai_answer, matched_intent, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_index,
                question,
                normalized_question,
                ai_answer,
                matched_intent,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()


def fetch_fallback_events(limit: int = 200) -> List[sqlite3.Row]:
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    with closing(conn) as conn2:
        cur = conn2.cursor()
        cur.execute(
            """
            SELECT * FROM fallback_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()


def fetch_sessions(limit: int = 100) -> List[sqlite3.Row]:
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    with closing(conn) as conn2:
        cur = conn2.cursor()
        cur.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()


# =========================================================
# 正規化 / 意図判定
# =========================================================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = t.replace("？", "").replace("?", "").replace("。", "").replace("、", "")
    t = re.sub(r"\s+", "", t)
    t = t.lower()
    for k, v in NORMALIZATION_MAP.items():
        t = t.replace(k, v)
    return t


@dataclass
class MatchResult:
    intent: Optional[str]
    answer: Optional[str]
    score: int


def match_intent(question: str) -> MatchResult:
    q = normalize_text(question)
    best_intent = None
    best_answer = None
    best_score = 0

    for intent, rule in INTENT_RULES.items():
        score = 0
        for pattern in rule["patterns"]:
            if re.search(pattern, q):
                score += 1
        if score > best_score:
            best_intent = intent
            best_answer = rule["answer"]
            best_score = score

    return MatchResult(best_intent, best_answer, best_score)


# =========================================================
# OpenAI
# =========================================================
def get_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai パッケージが見つかりません。 `pip install -U openai` を実行してください。")
    return OpenAI(api_key=api_key)


def compact_case_summary() -> str:
    return json.dumps(CASE_DATA, ensure_ascii=False, indent=2)


def llm_patient_answer(api_key: str, model_name: str, question: str, chat_history: List[Dict[str, str]]) -> str:
    client = get_client(api_key)
    case_text = compact_case_summary()
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-8:]])

    user_prompt = f"""
以下が症例情報です。
{case_text}

直近の会話:
{history_text}

学生からの質問:
{question}

この質問に対して、症例情報の範囲だけで患者として短く答えてください。
""".strip()

    # 互換性のため responses API → chat.completions API の順に試行
    try:
        response = client.responses.create(
            model=model_name,
            temperature=TEMPERATURE,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = getattr(response, "output_text", None)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    response = client.chat.completions.create(
        model=model_name,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# =========================================================
# 応答ロジック
# =========================================================
def fallback_safe_answer(question: str) -> str:
    q = normalize_text(question)
    generic_prompts = [
        "それについてもう少し詳しく教えてください",
        "詳しくはわかりません",
        "特に思い当たることはありません",
    ]
    if any(x in q for x in ["よろしく", "こんにちは", "こんばんは"]):
        return "よろしくお願いします。"
    if "名前" in q:
        return f"{CASE_DATA['patient']['name']}です。"
    if "年齢" in q or "おいくつ" in q:
        return f"{CASE_DATA['patient']['age']}歳です。"
    if "職業" in q or "お仕事" in q:
        return f"{CASE_DATA['patient']['occupation']}です。"
    return generic_prompts[0]


def generate_answer(question: str, api_key: str, model_name: str, chat_history: List[Dict[str, str]]) -> Tuple[str, str, str]:
    match = match_intent(question)
    normalized = normalize_text(question)

    # ルールで十分拾えた場合は即答
    if match.intent and match.score >= 1:
        return match.answer or "", "rule", match.intent

    # APIキーが無ければ安全な短文で返す
    if not api_key:
        return fallback_safe_answer(question), "safe_fallback", match.intent or ""

    # AI fallback
    try:
        answer = llm_patient_answer(api_key, model_name, question, chat_history)
        if not answer:
            answer = fallback_safe_answer(question)
        return answer, "ai", match.intent or ""
    except Exception as e:
        return f"すみません、少しうまく答えられませんでした。別の聞き方でお願いします。（詳細: {e}）", "error", match.intent or ""


# =========================================================
# 研究用集計
# =========================================================
def build_suggestion_json(events: List[sqlite3.Row]) -> str:
    grouped = defaultdict(list)
    for e in events:
        q = e["question"]
        nq = e["normalized_question"]
        if nq not in grouped["unmatched_questions"]:
            grouped["unmatched_questions"].append(nq)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_id": CASE_DATA["case_id"],
        "prompt_version": PROMPT_VERSION,
        "unmatched_questions": grouped["unmatched_questions"],
        "note": "この一覧を確認して、INTENT_RULES や NORMALIZATION_MAP に追加してください。",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def fallback_csv_bytes(events: List[sqlite3.Row]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "session_id", "turn_index", "question", "normalized_question", "ai_answer", "created_at"])
    for e in events:
        writer.writerow([e["id"], e["session_id"], e["turn_index"], e["question"], e["normalized_question"], e["ai_answer"], e["created_at"]])
    return buffer.getvalue().encode("utf-8-sig")


# =========================================================
# Streamlit UI
# =========================================================
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "turn_index" not in st.session_state:
        st.session_state.turn_index = 0
    if "started" not in st.session_state:
        st.session_state.started = False


def reset_chat(model_name: str):
    old_session_id = st.session_state.get("session_id")
    if old_session_id:
        close_session(old_session_id)

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "こんにちは。今日はどうされましたか。",
            "source": "system_opening",
            "intent": "chief_complaint",
        }
    ]
    st.session_state.session_id = create_session(model_name)
    st.session_state.turn_index = 1
    st.session_state.started = True

    save_turn(
        st.session_state.session_id,
        0,
        "assistant",
        "こんにちは。今日はどうされましたか。",
        "system_opening",
        "chief_complaint",
    )


def render_sidebar() -> Tuple[str, str, bool]:
    st.sidebar.title("設定")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model_name = st.sidebar.text_input("Model name", value=DEFAULT_MODEL_NAME)
    admin_mode = st.sidebar.checkbox("教員/研究モード", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**症例**: {CASE_DATA['title']}")
    st.sidebar.markdown(f"**患者**: {CASE_DATA['patient']['age']}歳 {CASE_DATA['patient']['sex']}")
    st.sidebar.markdown(f"**Prompt**: `{PROMPT_VERSION}`")

    if st.sidebar.button("新しい面接を開始"):
        reset_chat(model_name)

    if st.sidebar.button("面接を終了"):
        sid = st.session_state.get("session_id")
        if sid:
            close_session(sid)
        st.session_state.started = False
        st.sidebar.success("面接を終了しました。")

    return api_key, model_name, admin_mode


def render_case_info():
    with st.expander("症例の概要（教員確認用）", expanded=False):
        st.write(f"**主訴**: {CASE_DATA['chief_complaint']}")
        st.write(f"**現病歴**: {CASE_DATA['history_of_present_illness']['course']}")
        st.write(f"**既往歴**: {CASE_DATA['past_history']['medical']}")
        st.write(f"**内服**: {CASE_DATA['medications']}")
        st.write(f"**アレルギー**: {CASE_DATA['allergies']}")


def render_chat(api_key: str, model_name: str):
    st.subheader("模擬患者との面接")

    if not st.session_state.started or not st.session_state.session_id:
        st.info("左の「新しい面接を開始」を押してください。")
        return

    for msg in st.session_state.messages:
        avatar = "🧑‍⚕️" if msg["role"] == "user" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])
            if msg.get("role") == "assistant":
                source = msg.get("source", "")
                intent = msg.get("intent", "")
                badge = []
                if source:
                    badge.append(f"source: {source}")
                if intent:
                    badge.append(f"intent: {intent}")
                if badge:
                    st.caption(" / ".join(badge))

    question = st.chat_input("学生として質問を入力してください")
    if not question:
        return

    # user turn
    st.session_state.messages.append({"role": "user", "content": question, "source": "user", "intent": ""})
    save_turn(st.session_state.session_id, st.session_state.turn_index, "user", question, "user", "")

    # assistant turn
    answer, source, intent = generate_answer(question, api_key, model_name, st.session_state.messages)
    st.session_state.turn_index += 1

    st.session_state.messages.append({"role": "assistant", "content": answer, "source": source, "intent": intent})
    save_turn(st.session_state.session_id, st.session_state.turn_index, "assistant", answer, source, intent)

    if source == "ai":
        save_fallback_event(
            st.session_state.session_id,
            st.session_state.turn_index,
            question,
            normalize_text(question),
            answer,
            intent,
        )

    st.rerun()


def render_research_panel():
    st.subheader("研究・辞書改善パネル")
    events = fetch_fallback_events(limit=300)
    sessions = fetch_sessions(limit=100)

    total_sessions = len(sessions)
    total_fallbacks = len(events)
    fallback_by_question = Counter([e["normalized_question"] for e in events])
    top_questions = fallback_by_question.most_common(15)

    c1, c2, c3 = st.columns(3)
    c1.metric("保存セッション数", total_sessions)
    c2.metric("AI fallback数", total_fallbacks)
    c3.metric("ユニーク未対応質問", len(fallback_by_question))

    st.markdown("#### 未対応質問 上位")
    if top_questions:
        for q, n in top_questions:
            st.write(f"- {q}  × {n}")
    else:
        st.write("まだAI fallbackはありません。")

    st.markdown("#### 最新 fallback ログ")
    if events:
        for e in events[:30]:
            with st.expander(f"#{e['id']} | {e['created_at']} | {e['normalized_question']}"):
                st.write(f"**原文**: {e['question']}")
                st.write(f"**正規化後**: {e['normalized_question']}")
                st.write(f"**AI回答**: {e['ai_answer']}")
                if e["matched_intent"]:
                    st.write(f"**候補intent**: {e['matched_intent']}")
    else:
        st.write("ログはありません。")

    suggestion_json = build_suggestion_json(events)
    st.markdown("#### 辞書更新用 JSON 提案")
    st.code(suggestion_json, language="json")

    st.download_button(
        "fallbackログをCSVで保存",
        data=fallback_csv_bytes(events),
        file_name="fallback_events.csv",
        mime="text/csv",
    )
    st.download_button(
        "辞書更新候補JSONを保存",
        data=suggestion_json.encode("utf-8"),
        file_name="dictionary_update_suggestion.json",
        mime="application/json",
    )

    st.markdown("#### 現在のルール一覧")
    st.code(json.dumps({k: v["patterns"] for k, v in INTENT_RULES.items()}, ensure_ascii=False, indent=2), language="json")


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")
    st.title(APP_TITLE)
    st.caption("ルールベース優先 + AI fallback + 研究用ログ収集")

    init_db()
    init_state()

    api_key, model_name, admin_mode = render_sidebar()
    render_case_info()

    tabs = ["面接"]
    if admin_mode:
        tabs.append("研究パネル")

    selected = st.tabs(tabs)
    with selected[0]:
        render_chat(api_key, model_name)
    if admin_mode and len(selected) > 1:
        with selected[1]:
            render_research_panel()

    st.markdown("---")
    st.caption(
        "app7: 一般的な質問を優先的に辞書で処理し、未対応質問はAI fallbackとして記録します。"
    )


if __name__ == "__main__":
    main()
