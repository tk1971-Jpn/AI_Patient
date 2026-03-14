import json
import re
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st

# =========================================================
# 設定
# =========================================================
DB_PATH = "sessions.sqlite"
APP_TITLE = "AI模擬患者 - 救急外来問診トレーニング"
MODEL_NAME = "rule_based_patient_v1"
PROMPT_VERSION = "case_ac_chole_v1"
TEMPERATURE = 0.0


# =========================================================
# 症例データ
# =========================================================
CASE_DATA = {
    "case_id": "acute_cholecystitis_001",
    "chief_complaint": "右上腹部痛",
    "setting": "救急外来",
    "patient": {
        "name": "田中 和子",
        "age": 52,
        "sex": "女性",
        "occupation": "事務職",
    },
    "persona": {
        "tone": "ややつらそうだが受け答えは可能。聞かれたことに素直に答える。自分から長くは話しすぎない。",
        "style": "患者として自然に、簡潔に返答する。医学用語は使わない。聞かれていないことは必要以上に言わない。",
    },
    "ground_truth": {
        "present_illness": {
            "onset": "昨日の夜からです。",
            "course": "最初は胃のあたりが重い感じでしたが、だんだん右の上の方が痛くなってきました。",
            "location": "みぞおちから右の上腹部です。今は右上腹部が一番痛いです。",
            "character": "鈍い痛みというより、ズキズキして持続する感じです。",
            "severity": "かなり痛いです。10段階で7くらいです。",
            "duration": "昨日の夜からずっと続いています。",
            "progression": "少しずつ悪くなってきました。",
            "radiation": "背中の右側の方にも少しひびく感じがあります。",
            "trigger": "脂っこいものを食べたあとに悪くなった気がします。",
            "relieving": "横になっていてもあまり変わりません。",
            "aggravating": "動いたり深呼吸すると少し痛いです。",
            "associated_symptoms": {
                "fever": "熱っぽい感じがあって、今朝は少し熱がありました。",
                "nausea": "少し気持ち悪いです。",
                "vomiting": "吐いてはいません。",
                "diarrhea": "下痢はありません。",
                "constipation": "便秘は特にありません。",
                "appetite": "食欲はありません。",
                "jaundice": "自分では黄疸はわかりません。",
                "dark_urine": "尿の色は特に気になっていません。",
                "chest_pain": "胸の痛みはありません。",
                "dyspnea": "息苦しさはありません。",
            },
        },
        "past_history": {
            "medical": "高血圧があります。",
            "surgical": "手術はしたことがありません。",
            "gallstones": "健診で胆石があるかもと言われたことはありますが、詳しくは調べていません。",
        },
        "medications": "血圧の薬を飲んでいますが、名前はすぐには出てきません。",
        "allergies": "薬のアレルギーはありません。",
        "family_history": "家族に大きな病気の人はあまりいません。",
        "social_history": {
            "smoking": "たばこは吸いません。",
            "alcohol": "お酒はたまに少し飲むくらいです。",
            "living": "夫と2人暮らしです。",
        },
        "gynecologic": {
            "pregnancy": "妊娠はしていません。",
            "lmp": "閉経しています。",
        },
        "red_flags": {
            "syncope": "気を失ったことはありません。",
            "hematemesis": "吐血はありません。",
            "melena": "黒い便は出ていません。",
        },
        "vitals_hint": {
            "temperature": "熱は少しありそうです。",
            "general": "じっとしていてもつらそうです。",
        },
        "patient_concern": "何か悪い病気じゃないか心配です。胆石と関係あるんでしょうか。",
    },
    "ideal_summary_keywords": [
        "52歳",
        "女性",
        "右上腹部痛",
        "昨日夜から",
        "持続痛",
        "増悪",
        "発熱",
        "悪心",
        "食欲低下",
        "脂っこい食事後",
        "胆石",
        "高血圧",
        "急性胆嚢炎",
    ],
    "check_items": {
        "現病歴_発症時期": False,
        "現病歴_部位": False,
        "現病歴_性状": False,
        "現病歴_経過": False,
        "現病歴_増悪寛解因子": False,
        "随伴症状_発熱": False,
        "随伴症状_悪心嘔吐": False,
        "食事との関連": False,
        "既往歴": False,
        "内服薬": False,
        "アレルギー": False,
        "社会歴": False,
    },
}


# =========================================================
# DB
# =========================================================
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
        role TEXT,
        message TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        summary_text TEXT,
        auto_feedback TEXT,
        submitted_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_session_start(session_id: str, case_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO sessions (
        session_id, case_id, started_at, ended_at, model_name, prompt_version, temperature
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        case_id,
        datetime.now().isoformat(),
        None,
        MODEL_NAME,
        PROMPT_VERSION,
        TEMPERATURE
    ))
    conn.commit()
    conn.close()


def save_turn(session_id: str, role: str, message: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO turns (session_id, role, message, created_at)
    VALUES (?, ?, ?, ?)
    """, (
        session_id,
        role,
        message,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def save_summary(session_id: str, summary_text: str, auto_feedback: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO summaries (session_id, summary_text, auto_feedback, submitted_at)
    VALUES (?, ?, ?, ?)
    """, (
        session_id,
        summary_text,
        auto_feedback,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def end_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    UPDATE sessions
    SET ended_at = ?
    WHERE session_id = ?
    """, (datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()


# =========================================================
# テキスト処理
# =========================================================
def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("？", "?").replace("、", " ").replace("。", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


# =========================================================
# 問診チェック更新
# =========================================================
def update_check_items(user_text: str, check_items: Dict[str, bool]) -> Dict[str, bool]:
    t = normalize_text(user_text)

    if contains_any(t, ["いつから", "発症", "始まった", "何時から", "昨日から", "今朝から"]):
        check_items["現病歴_発症時期"] = True

    if contains_any(t, ["どこが痛", "場所", "部位", "右上腹部", "みぞおち", "お腹のどこ"]):
        check_items["現病歴_部位"] = True

    if contains_any(t, ["どんな痛み", "性状", "ズキズキ", "締め付け", "刺す", "重い感じ"]):
        check_items["現病歴_性状"] = True

    if contains_any(t, ["経過", "だんだん", "悪く", "良く", "変化", "続いて"]):
        check_items["現病歴_経過"] = True

    if contains_any(t, ["何で悪化", "楽になる", "増悪", "寛解", "体位", "動くと", "深呼吸", "食事で"]):
        check_items["現病歴_増悪寛解因子"] = True

    if contains_any(t, ["熱", "発熱", "何度"]):
        check_items["随伴症状_発熱"] = True

    if contains_any(t, ["吐き気", "悪心", "嘔吐", "吐いた"]):
        check_items["随伴症状_悪心嘔吐"] = True

    if contains_any(t, ["食事", "脂っこい", "食べたあと", "食後"]):
        check_items["食事との関連"] = True

    if contains_any(t, ["既往", "持病", "今までの病気", "高血圧", "糖尿病", "胆石"]):
        check_items["既往歴"] = True

    if contains_any(t, ["薬", "内服", "飲んでいる", "処方"]):
        check_items["内服薬"] = True

    if contains_any(t, ["アレルギー", "薬アレルギー", "食物アレルギー"]):
        check_items["アレルギー"] = True

    if contains_any(t, ["たばこ", "喫煙", "飲酒", "お酒", "家族", "仕事", "同居"]):
        check_items["社会歴"] = True

    return check_items


# =========================================================
# 患者応答ロジック（安定重視のルールベース）
# =========================================================
def generate_patient_response(user_text: str) -> str:
    t = normalize_text(user_text)
    gt = CASE_DATA["ground_truth"]

    # 開始直後の曖昧な質問
    if contains_any(t, ["どうしました", "今日はどうしました", "どのような症状", "主訴"]):
        return f"右の上のお腹が痛くて来ました。"

    # 発症時期
    if contains_any(t, ["いつから", "発症", "始まった", "何時から"]):
        return gt["present_illness"]["onset"]

    # 部位
    if contains_any(t, ["どこが痛", "場所", "部位", "どのあたり"]):
        return gt["present_illness"]["location"]

    # 性状
    if contains_any(t, ["どんな痛み", "性状", "痛みの感じ", "どのような痛み"]):
        return gt["present_illness"]["character"]

    # 痛みの強さ
    if contains_any(t, ["どれくらい痛い", "痛みの強さ", "10段階", "nrs"]):
        return gt["present_illness"]["severity"]

    # 持続時間・経過
    if contains_any(t, ["ずっと", "続いて", "持続", "経過", "悪く", "良くなって"]):
        return gt["present_illness"]["course"] + " " + gt["present_illness"]["progression"]

    # 放散
    if contains_any(t, ["ひびく", "放散", "背中", "肩に"]):
        return gt["present_illness"]["radiation"]

    # 食事関連
    if contains_any(t, ["食事", "脂っこい", "食後", "何か食べて"]):
        return gt["present_illness"]["trigger"]

    # 増悪寛解因子
    if contains_any(t, ["何で悪化", "何で楽", "深呼吸", "動くと", "体勢", "横になると"]):
        return gt["present_illness"]["aggravating"] + " " + gt["present_illness"]["relieving"]

    # 発熱
    if contains_any(t, ["熱", "発熱", "何度"]):
        return gt["present_illness"]["associated_symptoms"]["fever"]

    # 悪心
    if contains_any(t, ["吐き気", "悪心", "気持ち悪"]):
        return gt["present_illness"]["associated_symptoms"]["nausea"]

    # 嘔吐
    if contains_any(t, ["吐いた", "嘔吐"]):
        return gt["present_illness"]["associated_symptoms"]["vomiting"]

    # 下痢
    if contains_any(t, ["下痢"]):
        return gt["present_illness"]["associated_symptoms"]["diarrhea"]

    # 便秘
    if contains_any(t, ["便秘"]):
        return gt["present_illness"]["associated_symptoms"]["constipation"]

    # 食欲
    if contains_any(t, ["食欲", "食べられる", "食べれて"]):
        return gt["present_illness"]["associated_symptoms"]["appetite"]

    # 黄疸
    if contains_any(t, ["黄疸", "目が黄色", "皮膚が黄色"]):
        return gt["present_illness"]["associated_symptoms"]["jaundice"]

    # 胸痛
    if contains_any(t, ["胸痛", "胸が痛い"]):
        return gt["present_illness"]["associated_symptoms"]["chest_pain"]

    # 呼吸苦
    if contains_any(t, ["息苦しい", "呼吸苦", "息切れ"]):
        return gt["present_illness"]["associated_symptoms"]["dyspnea"]

    # 既往歴
    if contains_any(t, ["既往", "持病", "今までの病気", "病気はありますか"]):
        return gt["past_history"]["medical"] + " " + gt["past_history"]["gallstones"]

    # 手術歴
    if contains_any(t, ["手術したこと", "手術歴", "オペ"]):
        return gt["past_history"]["surgical"]

    # 内服
    if contains_any(t, ["薬", "内服", "飲んでいる薬"]):
        return gt["medications"]

    # アレルギー
    if contains_any(t, ["アレルギー", "薬アレルギー"]):
        return gt["allergies"]

    # 家族歴
    if contains_any(t, ["家族歴", "家族に病気", "ご家族に"]):
        return gt["family_history"]

    # 喫煙
    if contains_any(t, ["たばこ", "喫煙"]):
        return gt["social_history"]["smoking"]

    # 飲酒
    if contains_any(t, ["お酒", "飲酒", "アルコール"]):
        return gt["social_history"]["alcohol"]

    # 同居
    if contains_any(t, ["誰と住んで", "同居", "家族構成"]):
        return gt["social_history"]["living"]

    # 妊娠
    if contains_any(t, ["妊娠", "妊娠の可能性"]):
        return gt["gynecologic"]["pregnancy"]

    # 月経
    if contains_any(t, ["最終月経", "月経", "生理"]):
        return gt["gynecologic"]["lmp"]

    # 赤旗
    if contains_any(t, ["失神", "気を失", "吐血", "黒い便", "下血"]):
        answers = []
        if contains_any(t, ["失神", "気を失"]):
            answers.append(gt["red_flags"]["syncope"])
        if contains_any(t, ["吐血"]):
            answers.append(gt["red_flags"]["hematemesis"])
        if contains_any(t, ["黒い便", "下血"]):
            answers.append(gt["red_flags"]["melena"])
        return " ".join(answers)

    # 心配ごと
    if contains_any(t, ["心配", "不安", "どう思う", "気になっている"]):
        return gt["patient_concern"]

    # 名前
    if contains_any(t, ["お名前", "名前"]):
        return f"{CASE_DATA['patient']['name']}です。"

    # 年齢
    if contains_any(t, ["何歳", "年齢"]):
        return f"{CASE_DATA['patient']['age']}歳です。"

    # fallback
    return "すみません、もう少し具体的に聞いていただけますか。"


# =========================================================
# フィードバック
# =========================================================
def calc_interview_score(check_items: Dict[str, bool]) -> Tuple[int, List[str], List[str]]:
    total = len(check_items)
    achieved = sum(1 for v in check_items.values() if v)
    score = int(100 * achieved / total)

    done = [k for k, v in check_items.items() if v]
    missed = [k for k, v in check_items.items() if not v]

    return score, done, missed


def evaluate_summary(summary_text: str, ideal_keywords: List[str]) -> Dict:
    matched = []
    missing = []

    lower_summary = summary_text.lower()

    for kw in ideal_keywords:
        if kw.lower() in lower_summary:
            matched.append(kw)
        else:
            missing.append(kw)

    score = int(100 * len(matched) / len(ideal_keywords))

    if score >= 80:
        comment = "要点をよく押さえたサマリーです。"
    elif score >= 60:
        comment = "概ね重要点は含まれていますが、いくつか補足するとより良くなります。"
    else:
        comment = "重要な病歴要素の抜けがあります。現病歴・随伴症状・既往歴を整理して再確認すると良いです。"

    return {
        "score": score,
        "matched": matched,
        "missing": missing,
        "comment": comment
    }


def build_feedback_text(interview_score: int, done: List[str], missed: List[str], summary_eval: Dict) -> str:
    feedback = {
        "interview_score": interview_score,
        "obtained_items": done,
        "missed_items": missed,
        "summary_score": summary_eval["score"],
        "summary_matched": summary_eval["matched"],
        "summary_missing": summary_eval["missing"],
        "comment": summary_eval["comment"]
    }
    return json.dumps(feedback, ensure_ascii=False, indent=2)


# =========================================================
# セッション初期化
# =========================================================
def initialize_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        save_session_start(st.session_state.session_id, CASE_DATA["case_id"])

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    f"あなたは救急外来の患者です。"
                    f"「{CASE_DATA['patient']['name']}」さん、{CASE_DATA['patient']['age']}歳、{CASE_DATA['patient']['sex']}です。"
                    f" 学生からの質問に患者として答えてください。"
                )
            }
        ]

    if "check_items" not in st.session_state:
        st.session_state.check_items = CASE_DATA["check_items"].copy()

    if "summary_submitted" not in st.session_state:
        st.session_state.summary_submitted = False

    if "feedback_text" not in st.session_state:
        st.session_state.feedback_text = ""

    if "ended" not in st.session_state:
        st.session_state.ended = False


# =========================================================
# UI
# =========================================================
def render_sidebar():
    st.sidebar.header("症例情報")
    st.sidebar.write(f"**場面**: {CASE_DATA['setting']}")
    st.sidebar.write(f"**主訴**: {CASE_DATA['chief_complaint']}")
    st.sidebar.write("**学習目標**")
    st.sidebar.write("・腹痛患者への基本的な問診を行う")
    st.sidebar.write("・急性胆嚢炎を疑う情報を収集する")
    st.sidebar.write("・簡潔な症例サマリーを作成する")

    score, done, missed = calc_interview_score(st.session_state.check_items)
    st.sidebar.subheader("問診進捗")
    st.sidebar.progress(score / 100)
    st.sidebar.write(f"取得項目: {sum(st.session_state.check_items.values())}/{len(st.session_state.check_items)}")

    with st.sidebar.expander("取得できた項目"):
        for item in done:
            st.write(f"- {item}")

    with st.sidebar.expander("未取得項目"):
        for item in missed:
            st.write(f"- {item}")


def render_chat():
    st.subheader("自由問診")

    for msg in st.session_state.messages[1:]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

    if not st.session_state.summary_submitted:
        user_input = st.chat_input("患者さんに質問してください")
        if user_input:
            # user
            st.session_state.messages.append({"role": "user", "content": user_input})
            save_turn(st.session_state.session_id, "user", user_input)

            # update check items
            st.session_state.check_items = update_check_items(
                user_input,
                st.session_state.check_items
            )

            # patient response
            reply = generate_patient_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            save_turn(st.session_state.session_id, "assistant", reply)

            st.rerun()


def render_summary_section():
    st.markdown("---")
    st.subheader("症例サマリー提出")

    default_text = (
        "例）52歳女性。昨日夜からの右上腹部痛を主訴に救急外来を受診。"
        "痛みは持続性で増悪傾向。発熱、悪心、食欲低下あり。"
        "脂っこい食事後に悪化。高血圧あり。胆石を指摘された既往あり。"
        "急性胆嚢炎を疑う。"
    )

    summary_text = st.text_area(
        "患者情報をふまえて簡潔にまとめてください",
        value="",
        height=180,
        placeholder=default_text,
        disabled=st.session_state.summary_submitted
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("サマリーを提出", disabled=st.session_state.summary_submitted):
            if not summary_text.strip():
                st.warning("サマリーを入力してください。")
            else:
                interview_score, done, missed = calc_interview_score(st.session_state.check_items)
                summary_eval = evaluate_summary(summary_text, CASE_DATA["ideal_summary_keywords"])
                feedback_text = build_feedback_text(interview_score, done, missed, summary_eval)

                st.session_state.feedback_text = feedback_text
                st.session_state.summary_submitted = True

                save_summary(
                    st.session_state.session_id,
                    summary_text,
                    feedback_text
                )

                if not st.session_state.ended:
                    end_session(st.session_state.session_id)
                    st.session_state.ended = True

                st.rerun()

    with col2:
        if st.button("面接を終了して最初からやり直す"):
            if not st.session_state.ended:
                end_session(st.session_state.session_id)

            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def render_feedback():
    if st.session_state.summary_submitted and st.session_state.feedback_text:
        st.markdown("---")
        st.subheader("自動フィードバック")

        feedback = json.loads(st.session_state.feedback_text)

        st.write(f"**問診スコア**: {feedback['interview_score']} / 100")
        st.write(f"**サマリースコア**: {feedback['summary_score']} / 100")
        st.info(feedback["comment"])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**問診で取得できた項目**")
            for item in feedback["obtained_items"]:
                st.write(f"- {item}")

            st.markdown("**サマリーに含まれていた要点**")
            for item in feedback["summary_matched"]:
                st.write(f"- {item}")

        with col2:
            st.markdown("**追加で確認したい項目**")
            for item in feedback["missed_items"]:
                st.write(f"- {item}")

            st.markdown("**サマリーで補足したい要点**")
            for item in feedback["summary_missing"]:
                st.write(f"- {item}")

        with st.expander("保存されたフィードバックJSON"):
            st.code(st.session_state.feedback_text, language="json")


# =========================================================
# メイン
# =========================================================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")
    init_db()
    initialize_session_state()

    st.title(APP_TITLE)
    st.caption("急性胆嚢炎症例 / 学生向け / 安定性重視のルールベース模擬患者")

    render_sidebar()

    tab1, tab2 = st.tabs(["問診", "説明"])

    with tab1:
        render_chat()
        render_summary_section()
        render_feedback()

    with tab2:
        st.markdown("""
### このアプリの構造
- 患者応答は **ルールベース**
- 症例は **急性胆嚢炎1症例**
- 自由問診の内容と患者応答は **SQLiteに保存**
- 最後に学生が **症例サマリーを提出**
- 問診取得項目とサマリー内容を簡易評価

### 今後の拡張候補
1. 言い換え辞書をさらに増やして安定性向上  
2. 身体診察・検査結果提示モード追加  
3. 教員評価フォーム追加  
4. AIによる自由記述フィードバック追加  
5. ルールベース + LLM補助のハイブリッド化  
        """)


if __name__ == "__main__":
    main()