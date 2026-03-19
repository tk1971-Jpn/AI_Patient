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
APP_TITLE = "AI模擬患者 - OSCE医療面接トレーニング"
DEFAULT_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-mini")
PROMPT_VERSION = "app8_osce_flow_v1"
TEMPERATURE = 0.2

# ---------------------------------------------------------
# 症例設定（急性胆嚢炎）
# ---------------------------------------------------------
CASE_DATA = {
    "case_id": "acute_cholecystitis_001",
    "title": "OSCE：腹痛患者の医療面接（急性胆嚢炎）",
    "difficulty": "medical_student",
    "patient": {
        "name": "田中 恒一",
        "name_kana": "たなか こういち",
        "birth_date": "1973年4月18日",
        "age": 52,
        "sex": "男性",
        "occupation": "会社員",
        "marital_status": "既婚",
    },
    "chief_complaint": "右上腹部痛と発熱です。",
    "opening_statement": "昨日の夜から右のお腹が痛くて来ました。",
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
# OSCE冒頭フロー
# ---------------------------------------------------------
OSCE_STAGES = [
    "await_call",
    "await_self_intro",
    "await_identity_check",
    "await_open_question",
    "history_taking",
]

OSCE_HINTS = {
    "await_call": "患者さんを名前で呼び込みます。例：『田中恒一さん、どうぞお入りください。』",
    "await_self_intro": "自己紹介をします。例：『学生の木村です。本日担当します。』",
    "await_identity_check": "本人確認をします。氏名、必要に応じて生年月日を確認します。",
    "await_open_question": "オープンクエスチョンで開始します。例：『本日はどうされましたか。』",
    "history_taking": "通常の問診フェーズです。",
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
            r"今日はなぜ来られましたか",
            r"今日はなぜ受診",
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
    "今日はなぜ来られましたか": "本日はどうされましたか",
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
- OSCEの冒頭では、呼び込み・自己紹介・本人確認・オープンクエスチョンの流れを尊重する。
- オープンクエスチョン前に主訴を勝手に長く話し始めない。

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
                osce_stage TEXT,
                created_at TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS osce_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                turn_index INTEGER,
                event_type TEXT,
                stage_before TEXT,
                stage_after TEXT,
                user_text TEXT,
                detail TEXT,
                created_at TEXT
            )
            """
        )

        conn.commit()


def ensure_column_exists(table_name: str, column_name: str, column_type: str):
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cur.fetchall()]
        if column_name not in columns:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
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


def save_fallback_event(
    session_id: str,
    turn_index: int,
    question: str,
    normalized_question: str,
    ai_answer: str,
    matched_intent: str = "",
    osce_stage: str = "",
):
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO fallback_events (session_id, turn_index, question, normalized_question, ai_answer, matched_intent, osce_stage, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_index,
                question,
                normalized_question,
                ai_answer,
                matched_intent,
                osce_stage,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()


def save_osce_event(
    session_id: str,
    turn_index: int,
    event_type: str,
    stage_before: str,
    stage_after: str,
    user_text: str,
    detail: str,
):
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO osce_events (session_id, turn_index, event_type, stage_before, stage_after, user_text, detail, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_index,
                event_type,
                stage_before,
                stage_after,
                user_text,
                detail,
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
        cur.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,))
        return cur.fetchall()


def fetch_osce_events(limit: int = 300) -> List[sqlite3.Row]:
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    with closing(conn) as conn2:
        cur = conn2.cursor()
        cur.execute(
            """
            SELECT * FROM osce_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()


# =========================================================
# 正規化 / 意図判定 / OSCE判定
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


@dataclass
class OsceDetection:
    action: Optional[str]
    detail: str


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


def detect_osce_action(text: str) -> OsceDetection:
    q = normalize_text(text)
    patient_name = normalize_text(CASE_DATA["patient"]["name"])
    birth = normalize_text(CASE_DATA["patient"]["birth_date"])

    if patient_name and patient_name in q:
        if any(k in q for k in ["さん", "お入り", "どうぞ", "入って", "お呼び", "呼"]):
            return OsceDetection("call_patient", "名前で呼び込み")
        return OsceDetection("mentions_name", "氏名に言及")

    if any(k in q for k in ["私", "担当", "学生", "研修", "医師", "木村", "申します", "よろしくお願いします"]):
        if any(k in q for k in ["申します", "担当", "学生", "医師", "研修"]):
            return OsceDetection("self_intro", "自己紹介")

    if any(k in q for k in ["お名前", "氏名", "フルネーム", "生年月日", "誕生日"]):
        if "生年月日" in q or "誕生日" in q or birth in q:
            return OsceDetection("identity_birth", "本人確認（生年月日）")
        return OsceDetection("identity_name", "本人確認（氏名）")

    if any(re.search(p, q) for p in INTENT_RULES["chief_complaint"]["patterns"]):
        return OsceDetection("open_question", "オープンクエスチョン")

    return OsceDetection(None, "")


# =========================================================
# OpenAI
# =========================================================
def get_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai パッケージが見つかりません。 `pip install -U openai` を実行してください。")
    return OpenAI(api_key=api_key)


def compact_case_summary() -> str:
    return json.dumps(CASE_DATA, ensure_ascii=False, indent=2)


def llm_patient_answer(api_key: str, model_name: str, question: str, chat_history: List[Dict[str, str]], osce_stage: str) -> str:
    client = get_client(api_key)
    case_text = compact_case_summary()
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-8:]])

    user_prompt = f"""
以下が症例情報です。
{case_text}

現在のOSCE段階:
{osce_stage}

直近の会話:
{history_text}

学生からの質問:
{question}

この質問に対して、症例情報の範囲だけで患者として短く答えてください。
""".strip()

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
# OSCE冒頭応答
# =========================================================
def osce_stage_response(question: str, stage: str) -> Tuple[Optional[str], str, str, str, Optional[str]]:
    detection = detect_osce_action(question)
    stage_before = stage

    if stage == "await_call":
        if detection.action == "call_patient":
            return "はい。", "osce_rule", "call_patient", "await_self_intro", "呼び込みができています。"
        return "（患者はまだ呼び込まれていません。名前で呼び込んでください。）", "osce_coach", "await_call", stage_before, "呼び込み前です。"

    if stage == "await_self_intro":
        if detection.action == "self_intro":
            return "よろしくお願いします。", "osce_rule", "self_intro", "await_identity_check", "自己紹介ができています。"
        if detection.action == "call_patient":
            return "はい。", "osce_rule", "call_patient_repeat", stage_before, "呼び込みは済んでいます。次は自己紹介です。"
        return "お願いします。まずお名前を呼んでいただいて、自己紹介を聞かせてください。", "osce_coach", "await_self_intro", stage_before, "自己紹介待ちです。"

    if stage == "await_identity_check":
        if detection.action == "identity_name":
            return f"{CASE_DATA['patient']['name']}です。", "osce_rule", "identity_name", stage_before, "氏名確認ができています。必要なら生年月日も確認してください。"
        if detection.action == "identity_birth":
            return f"{CASE_DATA['patient']['birth_date']}です。", "osce_rule", "identity_birth", "await_open_question", "生年月日確認ができています。次はオープンクエスチョンです。"
        if detection.action == "open_question":
            return "その前に、確認のため名前と生年月日をお伝えした方がよいですか。", "osce_coach", "identity_needed", stage_before, "本人確認が先です。"
        return "田中 恒一です。必要でしたら生年月日も確認してください。", "osce_coach", "await_identity_check", stage_before, "本人確認待ちです。"

    if stage == "await_open_question":
        if detection.action == "open_question":
            return CASE_DATA["chief_complaint"], "osce_rule", "open_question", "history_taking", "オープンクエスチョンで開始できています。"
        if detection.action in {"identity_name", "identity_birth"}:
            return "はい。", "osce_rule", detection.action, stage_before, "本人確認は済んでいます。次はオープンクエスチョンです。"
        return "はい。", "osce_coach", "await_open_question", stage_before, "次は『本日はどうされましたか』のように始めてください。"

    return None, "", "", stage_before, None


# =========================================================
# 通常応答ロジック
# =========================================================
def fallback_safe_answer(question: str) -> str:
    q = normalize_text(question)
    if any(x in q for x in ["よろしく", "こんにちは", "こんばんは"]):
        return "よろしくお願いします。"
    if "名前" in q or "氏名" in q or "フルネーム" in q:
        return f"{CASE_DATA['patient']['name']}です。"
    if "生年月日" in q or "誕生日" in q:
        return f"{CASE_DATA['patient']['birth_date']}です。"
    if "年齢" in q or "おいくつ" in q:
        return f"{CASE_DATA['patient']['age']}歳です。"
    if "職業" in q or "お仕事" in q:
        return f"{CASE_DATA['patient']['occupation']}です。"
    return "それについてもう少し詳しく教えてください。"


def generate_answer(question: str, api_key: str, model_name: str, chat_history: List[Dict[str, str]], osce_stage: str) -> Tuple[str, str, str, str, Optional[str]]:
    # OSCE冒頭フェーズは専用処理
    if osce_stage != "history_taking":
        answer, source, intent, next_stage, note = osce_stage_response(question, osce_stage)
        if answer is not None:
            return answer, source, intent, next_stage, note

    match = match_intent(question)

    if match.intent and match.score >= 1:
        return match.answer or "", "rule", match.intent, "history_taking", None

    if not api_key:
        return fallback_safe_answer(question), "safe_fallback", match.intent or "", "history_taking", None

    try:
        answer = llm_patient_answer(api_key, model_name, question, chat_history, osce_stage)
        if not answer:
            answer = fallback_safe_answer(question)
        return answer, "ai", match.intent or "", "history_taking", None
    except Exception as e:
        return f"すみません、少しうまく答えられませんでした。別の聞き方でお願いします。（詳細: {e}）", "error", match.intent or "", "history_taking", None


# =========================================================
# 研究用集計
# =========================================================
def build_suggestion_json(events: List[sqlite3.Row]) -> str:
    grouped = defaultdict(list)
    for e in events:
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
    writer.writerow(["id", "session_id", "turn_index", "question", "normalized_question", "ai_answer", "matched_intent", "osce_stage", "created_at"])
    for e in events:
        writer.writerow([
            e["id"], e["session_id"], e["turn_index"], e["question"], e["normalized_question"],
            e["ai_answer"], e["matched_intent"], e["osce_stage"], e["created_at"]
        ])
    return buffer.getvalue().encode("utf-8-sig")


def osce_csv_bytes(events: List[sqlite3.Row]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "session_id", "turn_index", "event_type", "stage_before", "stage_after", "user_text", "detail", "created_at"])
    for e in events:
        writer.writerow([
            e["id"], e["session_id"], e["turn_index"], e["event_type"], e["stage_before"],
            e["stage_after"], e["user_text"], e["detail"], e["created_at"]
        ])
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
    if "osce_stage" not in st.session_state:
        st.session_state.osce_stage = "await_call"
    if "osce_notes" not in st.session_state:
        st.session_state.osce_notes = []


def reset_chat(model_name: str):
    old_session_id = st.session_state.get("session_id")
    if old_session_id:
        close_session(old_session_id)

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "（OSCE開始：患者はまだ入室していません。名前で呼び込んでください。）",
            "source": "system_opening",
            "intent": "await_call",
        }
    ]
    st.session_state.session_id = create_session(model_name)
    st.session_state.turn_index = 1
    st.session_state.started = True
    st.session_state.osce_stage = "await_call"
    st.session_state.osce_notes = []

    save_turn(
        st.session_state.session_id,
        0,
        "assistant",
        "（OSCE開始：患者はまだ入室していません。名前で呼び込んでください。）",
        "system_opening",
        "await_call",
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
    st.sidebar.markdown(f"**現在の段階**: `{st.session_state.get('osce_stage', 'await_call')}`")
    st.sidebar.info(OSCE_HINTS.get(st.session_state.get("osce_stage", "await_call"), ""))

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
        st.write(f"**導入時の想定訴え**: {CASE_DATA['opening_statement']}")
        st.write(f"**現病歴**: {CASE_DATA['history_of_present_illness']['course']}")
        st.write(f"**既往歴**: {CASE_DATA['past_history']['medical']}")
        st.write(f"**内服**: {CASE_DATA['medications']}")
        st.write(f"**アレルギー**: {CASE_DATA['allergies']}")


def render_chat(api_key: str, model_name: str):
    st.subheader("模擬患者との面接")

    if not st.session_state.started or not st.session_state.session_id:
        st.info("左の『新しい面接を開始』を押してください。")
        return

    stage = st.session_state.osce_stage
    st.caption(f"現在のOSCE段階: {stage} | {OSCE_HINTS.get(stage, '')}")

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

    question = st.chat_input("学生として発話を入力してください")
    if not question:
        return

    current_stage = st.session_state.osce_stage

    st.session_state.messages.append({"role": "user", "content": question, "source": "user", "intent": ""})
    save_turn(st.session_state.session_id, st.session_state.turn_index, "user", question, "user", current_stage)

    answer, source, intent, next_stage, osce_note = generate_answer(
        question,
        api_key,
        model_name,
        st.session_state.messages,
        current_stage,
    )
    st.session_state.turn_index += 1
    st.session_state.osce_stage = next_stage

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
            current_stage,
        )

    if source in {"osce_rule", "osce_coach"}:
        save_osce_event(
            st.session_state.session_id,
            st.session_state.turn_index,
            intent or source,
            current_stage,
            next_stage,
            question,
            osce_note or "",
        )
        if osce_note:
            st.session_state.osce_notes.append({
                "stage_before": current_stage,
                "stage_after": next_stage,
                "detail": osce_note,
                "user_text": question,
            })

    st.rerun()


def render_research_panel():
    st.subheader("研究・辞書改善パネル")
    events = fetch_fallback_events(limit=300)
    sessions = fetch_sessions(limit=100)
    osce_events = fetch_osce_events(limit=300)

    total_sessions = len(sessions)
    total_fallbacks = len(events)
    fallback_by_question = Counter([e["normalized_question"] for e in events])
    top_questions = fallback_by_question.most_common(15)
    osce_counter = Counter([e["event_type"] for e in osce_events])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("保存セッション数", total_sessions)
    c2.metric("AI fallback数", total_fallbacks)
    c3.metric("ユニーク未対応質問", len(fallback_by_question))
    c4.metric("OSCEイベント数", len(osce_events))

    st.markdown("#### OSCEイベント集計")
    if osce_counter:
        for k, n in osce_counter.most_common():
            st.write(f"- {k} × {n}")
    else:
        st.write("まだOSCEイベントはありません。")

    st.markdown("#### 未対応質問 上位")
    if top_questions:
        for q, n in top_questions:
            st.write(f"- {q} × {n}")
    else:
        st.write("まだAI fallbackはありません。")

    st.markdown("#### 最新OSCEログ")
    if osce_events:
        for e in osce_events[:30]:
            with st.expander(f"#{e['id']} | {e['created_at']} | {e['event_type']}"):
                st.write(f"**発話**: {e['user_text']}")
                st.write(f"**段階**: {e['stage_before']} → {e['stage_after']}")
                st.write(f"**詳細**: {e['detail']}")
    else:
        st.write("OSCEログはありません。")

    st.markdown("#### 最新 fallback ログ")
    if events:
        for e in events[:30]:
            with st.expander(f"#{e['id']} | {e['created_at']} | {e['normalized_question']}"):
                st.write(f"**原文**: {e['question']}")
                st.write(f"**正規化後**: {e['normalized_question']}")
                st.write(f"**AI回答**: {e['ai_answer']}")
                st.write(f"**OSCE段階**: {e['osce_stage']}")
                if e["matched_intent"]:
                    st.write(f"**候補intent**: {e['matched_intent']}")
    else:
        st.write("fallbackログはありません。")

    suggestion_json = build_suggestion_json(events)
    st.markdown("#### 辞書更新用 JSON 提案")
    st.code(suggestion_json, language="json")

    st.download_button(
        "fallbackログをCSVで保存",
        data=fallback_csv_bytes(events),
        file_name="fallback_events_app8.csv",
        mime="text/csv",
    )
    st.download_button(
        "OSCEログをCSVで保存",
        data=osce_csv_bytes(osce_events),
        file_name="osce_events_app8.csv",
        mime="text/csv",
    )
    st.download_button(
        "辞書更新候補JSONを保存",
        data=suggestion_json.encode("utf-8"),
        file_name="dictionary_update_suggestion_app8.json",
        mime="application/json",
    )

    st.markdown("#### 現在のルール一覧")
    st.code(json.dumps({k: v["patterns"] for k, v in INTENT_RULES.items()}, ensure_ascii=False, indent=2), language="json")


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")
    st.title(APP_TITLE)
    st.caption("OSCE冒頭フロー対応 + ルールベース優先 + AI fallback + 研究用ログ収集")

    init_db()
    ensure_column_exists("fallback_events", "osce_stage", "TEXT")
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
    st.caption("app8: OSCEの呼び込み・自己紹介・本人確認・オープンクエスチョンに対応した模擬患者アプリです。")


if __name__ == "__main__":
    main()
