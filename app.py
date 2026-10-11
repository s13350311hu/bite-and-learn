# -*- coding: utf-8 -*-
"""
Bite & Learn 跨界識食 - 智慧衛教 APP (V2)
"""
import os
import re
import io
import json
import html
import hashlib
import random
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
from ultralytics import YOLO

try:
    from gtts import gTTS
    GTTS_OK = True
except Exception:
    GTTS_OK = False

# ==========================================
# 0. 全域常數與 LLM 初始化
# ==========================================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ★ 修正 1：換成 1.5-flash 穩定版，免費額度較高，避免 429 錯誤
GEMINI_MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash")
llm_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

EXP_PER_QUIZ = 10
EXP_PER_LEVEL = 100
TZ_TAIPEI = timezone(timedelta(hours=8))

LANG_LEVELS = ["零基礎", "基礎", "進階"]
LEVEL_PROMPT_HINT = {
    "零基礎": "學習者完全沒學過該語言。句子必須極短，只用最基礎的單字。",
    "基礎": "學習者具備基礎文法與日常單字。句子 1~2 句，長度適中。",
    "進階": "學習者已有中高級能力。句子可包含較道地的慣用語或文化典故。",
}

# ==========================================
# 1. 頁面全域設定與 CSS
# ==========================================
st.set_page_config(page_title="Bite & Learn 跨界識食", page_icon="🥗", layout="wide")

st.markdown("""
<style>
    .card-box {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .card-header { font-size: 1.2rem; font-weight: 700; color: #0F172A; margin-bottom: 0.8rem; }
    .badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 9999px; font-weight: 600; }
    .badge-label { background-color: #EEF2FF; color: #4F46E5; }
    .abbr-sentence { font-size: 1.35rem; line-height: 2.1rem; padding: 0.6rem 0.2rem; }
    .abbr-sentence abbr { text-decoration: underline dotted #4F46E5; cursor: help; color: #4F46E5; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心資源載入與快取
# ==========================================
@st.cache_resource(show_spinner="正在載入 YOLOv8 視覺模型中...")
def load_yolo_model(model_name: str = "yolov8n.pt"):
    return YOLO(model_name)

@st.cache_data(show_spinner="載入食物衛教資料庫...")
def load_food_database(csv_path: str = "food_database.csv"):
    if not os.path.exists(csv_path):
        sample_data = {
            "AI_Label": ["pizza", "apple", "banana", "hot dog"],
            "EN_Word": ["Pizza", "Apple", "Banana", "Hot Dog"],
            "JP_Word": ["ピザ", "りんご", "バナナ", "ホットドッグ"],
            "Traffic_Light": ["🔴 紅燈", "🟢 綠燈", "🟡 黃燈", "🔴 紅燈"],
            "Bio_Warning": ["高鈉高脂注意！", "富含維他命C！", "運動後補給聖品！", "加工肉品請少吃！"],
            "Quiz_Question": ["披薩加鳳梨會惹怒哪國人？", "日本最有名的蘋果產地？", "香蕉富含哪種離子？", "熱狗名字由來跟哪種狗有關？"],
            "Quiz_Ans": ["義大利", "青森", "鉀", "臘腸狗"]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["AI_Label"] = df["AI_Label"].astype(str).str.strip().str.lower()
    return df

# ==========================================
# 3. Gemini API 呼叫層 (含快取機制)
# ==========================================
def _safe_generate(prompt: str, json_mode: bool = False):
    try:
        cfg = {"response_mime_type": "application/json"} if json_mode else None
        res = llm_model.generate_content(prompt, generation_config=cfg)
        return res.text, None
    except Exception as e:
        return None, e

def _parse_json(text: str):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned)

# ★ 修正 2：加入快取裝飾器，大幅節省 API 呼叫次數
@st.cache_data(show_spinner=False, ttl=3600)
def generate_dynamic_warning(food_name: str) -> str:
    prompt = f"你是一位具備醫學與營養學知識的大學生。使用者準備吃「{food_name}」。請用繁體中文，給予約 50 字的健康警告或營養提示，語氣生動活潑。"
    text, _ = _safe_generate(prompt)
    return text if text else "⚠️ API 暫時連線異常，請稍後再試。"

@st.cache_data(show_spinner=False, ttl=3600)
def generate_culture_text(en_word: str, jp_word: str, level: str) -> dict:
    prompt = f"""
    使用者辨識出食物：英文「{en_word}」、日文「{jp_word}」。學習者程度：{level}。{LEVEL_PROMPT_HINT[level]}
    請產生英日文生活化例句，並挑出 3 個單字附上中文意思。
    只回傳 JSON：
    {{"en": {{"sentence": "...", "translation_zh": "...", "vocab": [{{"word": "...", "meaning": "..."}}]}},
      "jp": {{"sentence": "...", "translation_zh": "...", "vocab": [{{"word": "...", "meaning": "..."}}]}}}}
    """
    raw, _ = _safe_generate(prompt, json_mode=True)
    try:
        return _parse_json(raw)
    except:
        return {
            "en": {"sentence": f"I eat {en_word}.", "translation_zh": f"我吃 {en_word}。", "vocab": []},
            "jp": {"sentence": f"私は{jp_word}を食べます。", "translation_zh": f"我吃 {jp_word}。", "vocab": []}
        }

@st.cache_data(show_spinner=False, ttl=3600)
def generate_quiz(en_word: str, level: str, seed: str) -> dict:
    prompt = f"""
    針對食物「{en_word}」出 1 道繁體中文四選一選擇題，難度：{level}。
    只回傳 JSON：
    {{"question": "...", "options": ["A選項", "B選項", "C選項", "D選項"], "answer": "必須與 options 其中一項完全相同", "explanation": "..."}}
    """
    raw, _ = _safe_generate(prompt, json_mode=True)
    try:
        data = _parse_json(raw)
        options = [str(o).strip() for o in data["options"]]
        rng = random.Random(seed)
        rng.shuffle(options)
        return {"question": data["question"], "options": options, "answer": data["answer"], "explanation": data.get("explanation", "")}
    except:
        return {"question": f"這是一道關於 {en_word} 的備用題目", "options": ["選項A", "選項B", "正解", "選項D"], "answer": "正解", "explanation": ""}

def build_abbr_html(sentence: str, vocab: list) -> str:
    mapping = {str(item.get("word", "")).strip().lower(): str(item.get("meaning", "")).strip() for item in vocab if isinstance(item, dict)}
    escaped_sentence = html.escape(sentence)
    if not mapping: return escaped_sentence
    words = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(html.escape(w)) for w in words) + r")(?![A-Za-z])", flags=re.IGNORECASE)
    def _wrap(match):
        original_text = match.group(1)
        meaning = html.escape(mapping.get(html.unescape(original_text).lower(), ""), quote=True)
        return f'<abbr title="{meaning}">{original_text}</abbr>' if meaning else original_text
    return pattern.sub(_wrap, escaped_sentence)

# ==========================================
# 4. 狀態管理核心
# ==========================================
def init_session_state():
    defaults = {
        "exp": 0, "streak_days": 1, "rewarded_ids": set(),
        "quiz_id": None, "quiz_data": None, "quiz_answered": False,
        "quiz_correct": None, "quiz_selected": None, "quiz_nonce": 0
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def reset_quiz_state():
    ss = st.session_state
    ss.quiz_answered = False
    ss.quiz_nonce += 1

def submit_answer():
    ss = st.session_state
    if ss.quiz_answered or ss.quiz_data is None: return
    choice = ss.get(f"quiz_choice_{ss.quiz_nonce}")
    if choice is None: return
    ss.quiz_answered = True
    ss.quiz_selected = choice
    ss.quiz_correct = (choice == ss.quiz_data["answer"])
    if ss.quiz_correct and ss.quiz_id not in ss.rewarded_ids:
        ss.rewarded_ids.add(ss.quiz_id)
        ss.exp += EXP_PER_QUIZ

def ensure_quiz(en_word: str, quiz_id: str, level: str):
    ss = st.session_state
    if ss.quiz_id != quiz_id:
        reset_quiz_state()
        ss.quiz_id = quiz_id
        with st.spinner("AI 出題老師準備中..."):
            ss.quiz_data = generate_quiz(en_word, level, seed=quiz_id)

# ==========================================
# 5. 側邊欄與分頁渲染
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.header("👤 玩家儀表板")
        st.metric("累積經驗值", f"{st.session_state.exp} EXP")
        st.divider()
        lang_level = st.selectbox("🌐 選擇語言程度", LANG_LEVELS, on_change=reset_quiz_state)
        input_method = st.radio("📥 輸入來源", ["📁 上傳圖片", "📷 開啟相機"], on_change=reset_quiz_state)
        img_data = None
        if "📁" in input_method:
            file = st.file_uploader("上傳食物照片", type=["jpg", "png", "webp"], on_change=reset_quiz_state)
            if file: img_data = file.getvalue()
        else:
            file = st.camera_input("拍攝食物照片", on_change=reset_quiz_state)
            if file: img_data = file.getvalue()
    return img_data, lang_level

def main():
    init_session_state()
    model = load_yolo_model()
    food_df = load_food_database()
    
    st.markdown("<h2>🥗 Bite & Learn 跨界識食</h2>", unsafe_allow_html=True)
    img_data, lang_level = render_sidebar()

    if not img_data:
        st.info("👈 請從左側上傳食物照片開始體驗！")
        return

    image = Image.open(io.BytesIO(img_data)).convert("RGB")
    image_sig = hashlib.md5(img_data).hexdigest()[:12]
    
    results = model.predict(source=image, conf=0.25, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.warning("畫面中找不到食物，請換張照片喔！")
        return

    best_idx = int(boxes.conf.argmax())
    detected_label = str(results[0].names[int(boxes.cls[best_idx].cpu().numpy())]).strip().lower()
    best_conf = float(boxes.conf[best_idx].cpu().numpy())
    plotted_rgb = results[0].plot()[..., ::-1]

    # ★ 修正 3：將語言程度加入鑰匙，切換難度會強迫 LLM 出新題目
    quiz_id = f"{image_sig}:{detected_label}:{lang_level}"

    matched = food_df[food_df["AI_Label"] == detected_label]
    if matched.empty:
        st.warning(f"AI 認出 {detected_label}，但不在資料庫中。")
        st.image(plotted_rgb, width=400)
        return

    row = matched.iloc[0]
    en_word, jp_word = row["EN_Word"], row["JP_Word"]

    tab_record, tab_lang, tab_quiz = st.tabs(["⚡ 快速健康紀錄", "🌍 外語文化探索", "🎯 每日測驗任務"])

    with tab_record:
        col1, col2 = st.columns(2)
        with col1:
            st.image(plotted_rgb, caption=f"標籤: {detected_label} | 信心度: {best_conf:.1%}")
        with col2:
            st.error(f"### {row['Traffic_Light']}")
            st.info(f"**生醫提醒**：\n{generate_dynamic_warning(en_word)}")

    with tab_lang:
        culture = generate_culture_text(en_word, jp_word, lang_level)
        st.markdown(f"**🇬🇧 英文 ({lang_level})**")
        st.markdown(f'<div class="abbr-sentence">{build_abbr_html(culture["en"]["sentence"], culture["en"]["vocab"])}</div>', unsafe_allow_html=True)
        st.markdown(f"**🇯🇵 日文 ({lang_level})**")
        st.markdown(f'<div class="abbr-sentence">{build_abbr_html(culture["jp"]["sentence"], culture["jp"]["vocab"])}</div>', unsafe_allow_html=True)

    with tab_quiz:
        ensure_quiz(en_word, quiz_id, lang_level)
        quiz = st.session_state.quiz_data
        st.markdown(f"#### ❓ {quiz['question']}")
        st.radio("選擇答案：", quiz["options"], key=f"quiz_choice_{st.session_state.quiz_nonce}", disabled=st.session_state.quiz_answered, index=None)
        st.button("✅ 送出答案", on_click=submit_answer, disabled=st.session_state.quiz_answered, type="primary")
        if st.session_state.quiz_answered:
            if st.session_state.quiz_correct: st.success("✅ 答對了！ EXP +10")
            else: st.error(f"❌ 答錯了～正解是：{quiz['answer']}")

if __name__ == "__main__":
    main()
