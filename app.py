
# -*- coding: utf-8 -*-
"""
Bite & Learn 跨界識食 - 智慧衛教 APP (V2)
======================================
技術棧架構：
  - 前端與互動：Streamlit (Tabs 分流、自訂卡片 CSS、遊戲化儀表板)
  - 電腦視覺：Ultralytics YOLOv8 (yolov8n.pt 預訓練模型)
  - 大型語言模型：Google Gemini (生醫警語 / 分級外語文本 / 動態測驗)
  - 資料處理：Pandas, Pillow (PIL), NumPy
 
V2 升級重點：
  1. st.tabs 介面分流
  2. 動態測驗 + 狀態鎖 (答對才加分、防刷分)
  3. 玩家儀表板 (Streak / EXP)
  4. 語言程度分級 (已實際串接 LLM Prompt)
  5. <abbr> 滑鼠懸停單字翻譯
 
【session_state 鍵值總覽】(維護者請先讀這段)
  --- 玩家進度 (整個 session 不重置) ---
  streak_days      : int   連續登入天數
  last_login_date  : str   上次登入日期 (YYYY-MM-DD, 台北時間)
  exp              : int   累積經驗值
  rewarded_ids     : set   已領過獎勵的「圖片+食物」ID (防止同一張圖重複刷分)
  --- 測驗狀態 (每次換圖都會被 reset_quiz_state() 清空) ---
  quiz_id          : str   目前測驗對應的「圖片簽章:食物標籤」
  quiz_data        : dict  目前題目 {question, options, answer, explanation}
  quiz_answered    : bool  【狀態鎖】True = 已作答，選項與按鈕全部鎖定
  quiz_correct     : bool  作答是否正確
  quiz_selected    : str   使用者送出的選項
  quiz_reward_msg  : str   獎勵提示文字
  quiz_warn        : bool  是否顯示「請先選擇答案」提示
  quiz_nonce       : int   radio 元件的版本號，改變它 = 讓 radio 變成全新元件 (清除舊選取)
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
 
# gTTS 為選用套件：裝了就用伺服器端語音 (st.audio)，沒裝則自動退回瀏覽器內建語音
try:
    from gtts import gTTS
    GTTS_OK = True
except Exception:
    GTTS_OK = False
 
# ==========================================
# 0. 全域常數與 LLM 初始化
# ==========================================
# 設定 API 金鑰 (存放於 .streamlit/secrets.toml 或 Streamlit Cloud Secrets)
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
 
# 模型名稱集中管理，日後換版只需改這一行
# 可在 secrets.toml 加一行 GEMINI_MODEL = "gemini-2.5-flash" 覆蓋，不用改程式碼
try:
    GEMINI_MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-3.8-flash")
except Exception:
    GEMINI_MODEL_NAME = "gemini-3.8-flash"
llm_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
 
# 遊戲化參數
EXP_PER_QUIZ = 10        # 答對一題獲得的 EXP
EXP_PER_LEVEL = 100      # 每升一級所需 EXP
TZ_TAIPEI = timezone(timedelta(hours=8))  # 以台北時間計算「今天」，避免雲端主機 UTC 造成日期錯位
 
# 語言程度分級 -> 對應的 LLM 提示語
LANG_LEVELS = ["零基礎", "基礎", "進階"]
LEVEL_PROMPT_HINT = {
    "零基礎": "學習者完全沒學過該語言。句子必須極短（英文 5~8 個字、日文 10 字內），只用最基礎的單字，日文請避免漢字以外的難詞。",
    "基礎": "學習者具備基礎文法與日常單字。句子 1~2 句，長度適中，可使用簡單的連接詞。",
    "進階": "學習者已有中高級能力。句子可包含較道地的慣用語、複合句或文化典故，詞彙可較豐富。",
}
 
# ==========================================
# 1. 頁面全域設定 (Page Config)
# ==========================================
st.set_page_config(
    page_title="Bite & Learn 跨界識食",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# 注入自訂 CSS 樣式提升 UI 現代卡片質感
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card-box {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .card-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-label {
        background-color: #EEF2FF;
        color: #4F46E5;
    }
    /* 滑鼠懸停翻譯：abbr 加上虛線底線與問號游標，提示使用者「可以懸停」 */
    .abbr-sentence {
        font-size: 1.35rem;
        line-height: 2.1rem;
        padding: 0.6rem 0.2rem;
    }
    .abbr-sentence abbr {
        text-decoration: underline dotted #4F46E5;
        text-underline-offset: 5px;
        cursor: help;
        color: #4F46E5;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)
 
 
# ==========================================
# 2. 核心資源載入與快取 (Cached Resources)
# ==========================================
@st.cache_resource(show_spinner="正在載入 YOLOv8 視覺模型中...")
def load_yolo_model(model_name: str = "yolov8n.pt") -> YOLO:
    """
    載入 Ultralytics YOLOv8 預訓練模型。
    使用 @st.cache_resource 避免重複讀取權重至記憶體。
    """
    model = YOLO(model_name)
    return model
 
 
@st.cache_data(show_spinner="載入食物衛教資料庫...")
def load_food_database(csv_path: str = "food_database.csv") -> pd.DataFrame:
    """
    讀取本地端的 food_database.csv 資料庫。
    若檔案不存在，會自動生成一份完整的示範資料庫供系統直接運行。
    """
    if not os.path.exists(csv_path):
        sample_data = {
            "AI_Label": [
                "pizza", "apple", "banana", "orange", "sandwich",
                "broccoli", "hot dog", "donut", "cake", "carrot"
            ],
            "EN_Word": [
                "Pizza", "Apple", "Banana", "Orange", "Sandwich",
                "Broccoli", "Hot Dog", "Donut", "Cake", "Carrot"
            ],
            "JP_Word": [
                "ピザ (Piza)", "りんご (Ringo)", "バナナ (Banana)", "オレンジ (Orenji)", "サンドイッチ (Sandoitchi)",
                "ブロッコリー (Burokkorī)", "ホットドッグ (Hottodoggu)", "ドーナツ (Dōnatsu)", "ケーキ (Kēki)", "にんじん (Ninjin)"
            ],
            "Traffic_Light": [
                "🔴 紅燈 (高鈉/高脂)", "🟢 綠燈 (高纖維)", "🟡 黃燈 (高鉀/天然糖份)", "🟢 綠燈 (高維他命C)",
                "🟡 黃燈 (碳水均衡/視夾餡)", "🟢 綠燈 (超抗氧化)", "🔴 紅燈 (高鈉/超加工肉品)",
                "🔴 紅燈 (超精製糖/反式脂肪)", "🔴 紅燈 (精製糖/高熱量)", "🟢 綠燈 (富含β-胡蘿蔔素)"
            ],
            "Bio_Warning": [
                "警告：明天早上臉絕對會水腫！高GI與飽和脂肪會引起餐後嗜睡與血糖劇烈波動。",
                "富含槲皮素與維他命C，抗氧化並幫助熬夜恢復氣色！果皮膳食纖維更有助於腸胃蠕動。",
                "運動後補給聖品！但高血壓腎功能不佳者需注意鉀離子代謝攝取量。",
                "柑橘類富含黃酮類與維生素C，能增強免疫力，但空腹過量食用易刺激胃酸。",
                "建議避開油炸肉排與大量美乃滋抹醬，選擇全麥麵包與水煮蛋更健康。",
                "十字花科蔬菜之王！富含蘿蔔硫素，具極佳的細胞保護與抗癌抗發炎效能。",
                "加工肉品含亞硝酸鹽與過量鈉，世界衛生組織列為一級致癌物，應限制食用頻率。",
                "高溫油炸與高糖霜結合，容易誘發體內慢性發炎及胰島素阻抗，切忌常吃！",
                "高精製糖與奶油熱量炸彈，易造成糖化終產物 (AGEs) 累積，加速皮膚老化。",
                "脂溶性維生素的最佳來源，建議搭配健康油脂烹調以促進吸收，有助眼睛保健。"
            ],
            "Quiz_Question": [
                "在義大利，披薩上加什麼會觸怒當地人？",
                "日本最有名的蘋果產地在哪個縣？",
                "香蕉彎曲生長的主要原因是什麼？",
                "橙色柑橘中主要提供亮麗橘色的抗氧化色素是什麼？",
                "三明治相傳是由哪一國的「三明治伯爵」所發明的？",
                "青花菜 (Broccoli) 與花椰菜 (Cauliflower) 其實是同一物種嗎？",
                "熱狗的名字由來傳說與哪一種狗的外型有關？",
                "為什麼傳統甜甜圈中間通常會有一個洞？",
                "法式經典「瑪德蓮蛋糕」著名的貝殼外型是由哪種容器烤出來的？",
                "兔子在自然環境中最常吃的食物其實是胡蘿蔔嗎？"
            ],
            "Quiz_Ans": [
                "鳳梨 (夏威夷披薩)",
                "青森縣",
                "背地性（向陽性，朝陽光方向生長）",
                "胡蘿蔔素與類黃酮",
                "英國 (John Montagu, 4th Earl of Sandwich)",
                "是的，兩者都是甘藍的變種",
                "臘腸犬 (Dachshund)",
                "為了讓油炸時受熱均勻，避免中心炸不熟",
                "貝殼形模具",
                "不是，野兔主要吃草和葉子，胡蘿蔔糖分過高反而不能多吃"
            ]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
 
    # 清理 AI_Label 欄位：去除首尾空白並轉小寫，確保字串比對精確無誤
    df["AI_Label"] = df["AI_Label"].astype(str).str.strip().str.lower()
    return df
 
 
# ==========================================
# 3. Gemini 呼叫層 (快取 + 容錯)
# ==========================================
@st.cache_data(show_spinner=False, ttl=3600)
def _gemini_generate(prompt: str, json_mode: bool = False) -> str:
    """
    最底層的 Gemini 呼叫。
    ★ 重要：這裡「不」捕捉例外，讓錯誤直接往外拋。
      因為 st.cache_data 只會快取「成功回傳」的結果；
      若在這裡把錯誤訊息當成字串 return，429 之類的暫時性錯誤會被快取住，
      使用者之後即使 API 恢復也會一直看到錯誤訊息。
    """
    cfg = {"response_mime_type": "application/json"} if json_mode else None
    response = llm_model.generate_content(prompt, generation_config=cfg)
    return response.text
 
 
def _friendly_error(e: Exception) -> str:
    """把例外轉成使用者看得懂的提示"""
    msg = str(e)
    if "429" in msg:
        return "⚠️ 哎呀！大家太熱情了，AI 護理師有點喘不過氣，請等待 10 秒後再試一次喔！"
    return f"⚠️ 發生未知異常：{msg}"
 
 
def _parse_json(text: str):
    """容錯解析 LLM 回傳的 JSON（會自動剝除 ```json 圍欄）"""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned)
 
 
def _log_llm_error(tag: str, e: Exception):
    """把真正的錯誤原因記下來，顯示在側邊欄「API 診斷」，不再被備援機制默默吞掉"""
    errs = st.session_state.setdefault("llm_errors", [])
    errs.append(f"[{datetime.now(TZ_TAIPEI):%H:%M:%S}] {tag} -> {type(e).__name__}: {e}")
    del errs[:-6]   # 只保留最近 6 筆
 
 
def _safe_generate(prompt: str, tag: str, json_mode: bool = False):
    """
    呼叫 Gemini，回傳 (文字, 例外)。
    若 JSON 模式失敗，會自動退回一般文字模式再試一次
    (部分模型/SDK 版本不支援 response_mime_type)。
    """
    try:
        return _gemini_generate(prompt, json_mode), None
    except Exception as e:
        _log_llm_error(tag, e)
        if json_mode:
            try:
                return _gemini_generate(prompt, False), None
            except Exception as e2:
                _log_llm_error(tag + "(retry)", e2)
                return None, e2
        return None, e
 
 
def generate_dynamic_warning(food_name: str) -> str:
    """將辨識出的食物名稱丟給 Gemini，動態生成生醫警語"""
    prompt = f"""
    你現在是一位充滿幽默感、具備醫學與營養學知識的生醫系大學生。
    使用者剛剛用系統掃描到準備吃「{food_name}」。
    請用繁體中文，用大約 50 到 80 字的一小段話，給予健康警告或營養提示。
    語氣要生動活潑、有點像在吐槽或關心朋友，讓大學生看了會有共鳴。
    """
    text, err = _safe_generate(prompt, "生醫警語")
    return text if text else _friendly_error(err)
 
 
# ---------- 外語文本 (依程度分級) ----------
def _fallback_culture(en_word: str, jp_word: str) -> dict:
    """LLM 失敗時的備援文本，確保外語頁籤永遠有內容可顯示"""
    jp_plain = jp_word.split(" (")[0].strip()
    return {
        "en": {
            "sentence": f"I want to eat {en_word} today.",
            "translation_zh": f"我今天想吃 {en_word}。",
            "vocab": [
                {"word": "eat", "meaning": "吃"},
                {"word": "today", "meaning": "今天"},
                {"word": en_word, "meaning": "（本次辨識的食物）"},
            ],
        },
        "jp": {
            "sentence": f"今日は{jp_plain}を食べたいです。",
            "translation_zh": f"我今天想吃{jp_plain}。",
            "vocab": [
                {"word": "今日", "meaning": "今天 (きょう)"},
                {"word": "食べたい", "meaning": "想吃 (たべたい)"},
                {"word": jp_plain, "meaning": "（本次辨識的食物）"},
            ],
        },
    }
 
 
def _valid_culture(data) -> bool:
    """檢查 LLM 回傳的 JSON 結構是否符合預期，避免畫面渲染時才爆錯"""
    try:
        for lang in ("en", "jp"):
            block = data[lang]
            if not isinstance(block["sentence"], str) or not block["sentence"].strip():
                return False
            if not isinstance(block["translation_zh"], str):
                return False
            if not isinstance(block["vocab"], list):
                return False
        return True
    except Exception:
        return False
 
 
def generate_culture_text(en_word: str, jp_word: str, level: str) -> tuple:
    """
    依語言程度 (level) 生成英/日文例句 + 中文翻譯 + 單字表。
    回傳 (資料 dict, 是否使用備援 bool)
    """
    prompt = f"""
    你是一位專業的外語教師。使用者剛辨識出食物：英文「{en_word}」、日文「{jp_word}」。
    學習者程度：{level}。{LEVEL_PROMPT_HINT[level]}
    請產生與這個食物相關的英文與日文各一段生活化例句（可帶一點飲食文化趣味），
    並挑出句子中 3~6 個值得學習的單字，附上繁體中文意思。
    ★ vocab 裡的 "word" 必須「逐字出現」在對應的 sentence 裡（大小寫一致）。
    只回傳 JSON，格式如下：
    {{
      "en": {{"sentence": "...", "translation_zh": "...", "vocab": [{{"word": "...", "meaning": "..."}}]}},
      "jp": {{"sentence": "...", "translation_zh": "...", "vocab": [{{"word": "...", "meaning": "..."}}]}}
    }}
    """
    raw, _err = _safe_generate(prompt, "外語例句", json_mode=True)
    if raw:
        try:
            data = _parse_json(raw)
            if _valid_culture(data):
                return data, False
            _log_llm_error("外語例句", ValueError("JSON 結構不符預期"))
        except Exception as e:
            _log_llm_error("外語例句(解析)", e)
    return _fallback_culture(en_word, jp_word), True
 
 
def build_abbr_html(sentence: str, vocab: list) -> str:
    """
    把句子中的單字包成 <abbr title="翻譯">單字</abbr>，滑鼠懸停即顯示翻譯。
 
    安全設計（因為後面要用 unsafe_allow_html=True 渲染 LLM 的輸出）：
      1. 先對整句 html.escape，LLM 若夾帶 <script> 之類標籤會被轉義成純文字
      2. 用「單次 re.sub + 單一合併 pattern」做替換，
         避免第二個單字意外比對到第一個 <abbr title="..."> 屬性裡的文字
    """
    mapping = {}
    for item in vocab:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word", "")).strip()
        meaning = str(item.get("meaning", "")).strip()
        if word and meaning:
            mapping[word.lower()] = meaning
 
    escaped_sentence = html.escape(sentence)
    if not mapping:
        return escaped_sentence
 
    # 長字優先比對 (例如 "hot dog" 要先於 "hot")；前後不可緊鄰英文字母，避免 eat 比對到 great
    words = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z])(" + "|".join(re.escape(html.escape(w)) for w in words) + r")(?![A-Za-z])",
        flags=re.IGNORECASE,
    )
 
    def _wrap(match):
        original_text = match.group(1)                      # 已是 escape 過的原文
        key = html.unescape(original_text).lower()
        meaning = html.escape(mapping.get(key, ""), quote=True)
        return f'<abbr title="{meaning}">{original_text}</abbr>' if meaning else original_text
 
    return pattern.sub(_wrap, escaped_sentence)
 
 
# ---------- 發音 (TTS) ----------
def clean_speech_text(text: str) -> str:
    """去掉『ピザ (Piza)』這類括號羅馬拼音，只留要念的文字"""
    return re.sub(r"\s*[\(（].*?[\)）]", "", str(text)).strip()
 
 
@st.cache_data(show_spinner=False, ttl=86400)
def tts_audio_bytes(text: str, lang: str) -> bytes:
    """gTTS 伺服器端語音 (需安裝 gTTS 並能連外)。失敗時讓例外往外拋，不快取錯誤。"""
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    return buf.getvalue()
 
 
def render_speech_panel(items: list, lang_code: str, height: int = 70):
    """
    以瀏覽器內建 Web Speech API 產生一排發音按鈕 (免 API、免安裝套件)。
    items: [(按鈕文字, 要念的文字), ...]；lang_code: 'en-US' / 'ja-JP'
    """
    buttons = "".join(
        f'<button onclick=\'speak({json.dumps(txt, ensure_ascii=False)})\'>🔊 {html.escape(label)}</button>'
        for label, txt in items if txt
    )
    page = f"""
    <style>
      button {{margin:2px 4px 2px 0;padding:4px 10px;border:1px solid #CBD5E1;border-radius:9999px;
              background:#EEF2FF;color:#4F46E5;font-size:14px;cursor:pointer;}}
      button:hover {{background:#E0E7FF;}}
    </style>
    <div>{buttons}</div>
    <script>
      function speak(t) {{
        if (!('speechSynthesis' in window)) {{ alert('此瀏覽器不支援語音合成'); return; }}
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(t);
        u.lang = '{lang_code}'; u.rate = 0.9;
        window.speechSynthesis.speak(u);
      }}
    </script>
    """
    components.html(page, height=height, scrolling=True)
 
 
def render_tts_audio(text: str, gtts_lang: str):
    """若有安裝 gTTS，額外提供可播放的音訊列；沒有就靜默略過"""
    if not GTTS_OK or not text:
        return
    try:
        st.audio(tts_audio_bytes(text, gtts_lang), format="audio/mp3")
    except Exception as e:
        _log_llm_error("gTTS", e)
 
 
# ---------- 動態測驗 ----------
def _fallback_quiz(row: pd.Series, food_df: pd.DataFrame, seed: str) -> dict:
    """
    LLM 失敗時的備援：用 CSV 的題目與正解，
    再從其他食物的正解裡抽 3 個當作干擾選項。
    """
    correct = str(row["Quiz_Ans"])
    pool = [str(a) for a in food_df["Quiz_Ans"].tolist() if str(a) != correct]
    rng = random.Random(seed)
    distractors = rng.sample(pool, k=min(3, len(pool)))
    options = distractors + [correct]
    rng.shuffle(options)
    return {
        "question": str(row["Quiz_Question"]),
        "options": options,
        "answer": correct,
        "explanation": "（本題來自內建文化題庫）",
    }
 
 
def generate_quiz(row: pd.Series, food_df: pd.DataFrame, level: str, seed: str) -> dict:
    """
    根據 YOLO 辨識出的食物，請 Gemini 動態出 1 道四選一選擇題。
    ★ 出題結果會被存進 session_state['quiz_data']，
      重新整理畫面 (rerun) 時直接讀取，不會每次都重新出題或重新洗牌。
    """
    en_word = row["EN_Word"]
    prompt = f"""
    你是一位出題老師。請針對食物「{en_word}」出 1 道繁體中文四選一選擇題，
    主題可以是營養知識、飲食文化或英日文單字，難度對應學習者程度：{level}。
    只回傳 JSON：
    {{"question": "...", "options": ["A選項", "B選項", "C選項", "D選項"], "answer": "必須與 options 其中一項完全相同", "explanation": "30 字內的解說"}}
    """
    raw, _err = _safe_generate(prompt, "動態測驗", json_mode=True)
    try:
        data = _parse_json(raw)
        options = [str(o).strip() for o in data["options"]]
        answer = str(data["answer"]).strip()
        # 驗證：4 個互異選項、且正解必須在選項中
        if len(options) == 4 and len(set(options)) == 4 and answer in options:
            rng = random.Random(seed)          # 以 seed 洗牌，確保同一題選項順序固定
            rng.shuffle(options)
            return {
                "question": str(data["question"]).strip(),
                "options": options,
                "answer": answer,
                "explanation": str(data.get("explanation", "")).strip(),
            }
    except Exception:
        pass
    return _fallback_quiz(row, food_df, seed)
 
 
# ==========================================
# 4. 狀態管理核心 (Session State & Callbacks)
# ==========================================
def today_str() -> str:
    return datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d")
 
 
def init_session_state():
    """
    初始化所有 session_state 鍵值（只在鍵不存在時設定，rerun 不會覆蓋既有進度）。
    並在這裡處理「連續登入天數 (Streak)」。
    """
    defaults = {
        "streak_days": 0,
        "last_login_date": None,
        "exp": 0,
        "rewarded_ids": set(),
        # --- 測驗狀態 ---
        "quiz_id": None,
        "quiz_data": None,
        "quiz_answered": False,
        "quiz_correct": None,
        "quiz_selected": None,
        "quiz_reward_msg": "",
        "quiz_warn": False,
        "quiz_nonce": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
 
    # ---- Streak 邏輯（可重複執行，結果一致 → 每次 rerun 呼叫也安全）----
    today = today_str()
    last = st.session_state.last_login_date
    if last != today:
        yesterday = (datetime.now(TZ_TAIPEI) - timedelta(days=1)).strftime("%Y-%m-%d")
        if last == yesterday:
            st.session_state.streak_days += 1     # 昨天有登入 -> 連續天數 +1
        else:
            st.session_state.streak_days = 1      # 首次登入或中斷過 -> 從 1 開始
        st.session_state.last_login_date = today
 
 
def reset_quiz_state():
    """
    【Callback】重置測驗狀態鎖。
    綁定在 st.file_uploader / st.camera_input / 輸入方式 radio 的 on_change。
 
    為什麼要用 callback 而不是在主程式 if 判斷？
      Callback 會在「下一次 rerun 開始之前」執行，
      所以主程式與側邊欄渲染時，讀到的一定是已重置的乾淨狀態，
      不會出現「新圖 + 舊的已作答鎖」的瞬間錯亂。
 
    注意：exp / streak / rewarded_ids 屬於玩家進度，這裡「刻意不重置」。
    """
    ss = st.session_state
    ss.quiz_id = None
    ss.quiz_data = None
    ss.quiz_answered = False
    ss.quiz_correct = None
    ss.quiz_selected = None
    ss.quiz_reward_msg = ""
    ss.quiz_warn = False
    # 把 radio 的 key 換成新版本號 -> Streamlit 視為全新元件，舊的選取值自動消失
    ss.quiz_nonce += 1
 
 
def submit_answer():
    """
    【Callback】送出答案。所有「改分數、上鎖」的動作都集中在這裡，
    不寫在主程式裡，原因：
      1. Callback 先於 rerun 執行，側邊欄的 EXP 會立刻顯示最新分數
      2. 主程式每次 rerun 都會重跑，若把加分寫在主程式會被重複觸發
    """
    ss = st.session_state
 
    # 【防刷分關卡 1】已作答就直接返回（連點、多分頁連送都會被擋下）
    if ss.quiz_answered or ss.quiz_data is None:
        return
 
    choice = ss.get(f"quiz_choice_{ss.quiz_nonce}")
    if choice is None:
        ss.quiz_warn = True          # 尚未選擇 -> 不上鎖，只提示
        return
 
    # 先上鎖，再做後續動作
    ss.quiz_warn = False
    ss.quiz_answered = True
    ss.quiz_selected = choice
    ss.quiz_correct = (choice == ss.quiz_data["answer"])
 
    if ss.quiz_correct:
        # 【防刷分關卡 2】同一張圖(同一個 quiz_id)只能領一次獎勵，
        # 即使使用者重新上傳同一張照片讓測驗被 reset，也無法再次加分
        if ss.quiz_id in ss.rewarded_ids:
            ss.quiz_reward_msg = "這張照片的獎勵已領取過囉，換一道菜再來挑戰吧！"
        else:
            ss.rewarded_ids.add(ss.quiz_id)
            ss.exp += EXP_PER_QUIZ
            ss.quiz_reward_msg = f"🎉 獲得 +{EXP_PER_QUIZ} EXP！"
    else:
        ss.quiz_reward_msg = ""
 
 
def ensure_quiz(row: pd.Series, food_df: pd.DataFrame, quiz_id: str, level: str):
    """
    確保 session_state 裡存的題目與「目前這張圖」一致。
    雙重保險：就算 on_change 沒被觸發（例如相機重拍），
    只要偵測到 quiz_id 不同，也會自動重置並重新出題。
    """
    ss = st.session_state
    if ss.quiz_id != quiz_id:
        reset_quiz_state()
        ss.quiz_id = quiz_id
        with st.spinner("AI 出題老師正在為你準備題目..."):
            ss.quiz_data = generate_quiz(row, food_df, level, seed=quiz_id)
 
 
# ==========================================
# 5. 側邊欄互動模組 (Sidebar)
# ==========================================
def render_player_dashboard():
    """玩家儀表板：連續登入天數 + EXP 進度條"""
    ss = st.session_state
    level_no = ss.exp // EXP_PER_LEVEL + 1
    exp_in_level = ss.exp % EXP_PER_LEVEL
 
    st.subheader("🎮 玩家儀表板")
    c1, c2 = st.columns(2)
    c1.metric("🔥 連續登入", f"{ss.streak_days} 天")
    c2.metric("⭐ 等級", f"Lv.{level_no}")
    st.progress(
        exp_in_level / EXP_PER_LEVEL,
        text=f"EXP {exp_in_level} / {EXP_PER_LEVEL}（累積 {ss.exp}）"
    )
 
 
def render_api_diagnostics():
    """側邊欄「API 診斷」：直接顯示 Gemini 真正的錯誤原因，不再只看到備援內容"""
    with st.expander("🔧 API 診斷"):
        st.caption(f"模型：`{GEMINI_MODEL_NAME}`｜gTTS：{'已安裝' if GTTS_OK else '未安裝（使用瀏覽器語音）'}")
        if st.button("測試 Gemini 連線", key="api_ping"):
            try:
                r = llm_model.generate_content("請只回覆：OK")
                st.success(f"連線成功：{r.text.strip()[:40]}")
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        if st.button("列出可用模型", key="api_list"):
            try:
                names = [m.name.replace("models/", "") for m in genai.list_models()
                         if "generateContent" in m.supported_generation_methods]
                st.code("\n".join(names) or "（沒有可用模型）")
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        for line in reversed(st.session_state.get("llm_errors", [])):
            st.caption(line)
 
 
def render_sidebar():
    """
    回傳 (image, image_sig, lang_level)
      image      : PIL 圖片或 None
      image_sig  : 圖片內容雜湊簽章，用來辨識「是不是同一張圖」
      lang_level : 使用者選擇的語言程度
    """
    with st.sidebar:
        st.title("🥗 Bite & Learn 跨界識食")
        st.markdown(
            """
            **跨領域智慧衛教系統**
            結合 **AI 電腦視覺**、**食品科學紅綠燈**、**生醫風險警語** 與 **外語文化微學習**。
            """
        )
        st.markdown("---")
 
        # ---- 玩家儀表板 ----
        render_player_dashboard()
        st.markdown("---")
 
        # ---- 語言程度分級（變數會傳入 LLM Prompt）----
        st.subheader("🌐 語言程度")
        lang_level = st.selectbox(
            label="選擇你的外語程度",
            options=LANG_LEVELS,
            index=0,
            key="lang_level",
            help="會影響外語例句與測驗題的難度"
        )
        st.markdown("---")
 
        # ---- 影像輸入 ----
        st.subheader("📥 選擇影像輸入方式")
        input_method = st.radio(
            label="請選擇輸入來源：",
            options=["📁 上傳圖片 (File Uploader)", "📷 開啟相機 (Camera Input)"],
            index=0,
            key="input_method",
            on_change=reset_quiz_state,       # 切換來源也重置測驗
        )
 
        input_image, image_sig = None, None
        if "📁 上傳圖片" in input_method:
            uploaded_file = st.file_uploader(
                label="選擇一張食物照片",
                type=["jpg", "jpeg", "png", "webp"],
                help="支援常見格式如 JPG、PNG、WEBP",
                key="uploader_widget",
                on_change=reset_quiz_state,   # ★ 上傳新圖 -> 測驗狀態鎖重置
            )
            if uploaded_file is not None:
                data = uploaded_file.getvalue()
                image_sig = hashlib.md5(data).hexdigest()[:12]
                input_image = Image.open(io.BytesIO(data))
        else:
            camera_file = st.camera_input(
                label="對準食物拍攝照片",
                help="需允許瀏覽器存取相機進行拍攝",
                key="camera_widget",
                on_change=reset_quiz_state,   # ★ 重新拍照 -> 測驗狀態鎖重置
            )
            if camera_file is not None:
                data = camera_file.getvalue()
                image_sig = hashlib.md5(data).hexdigest()[:12]
                input_image = Image.open(io.BytesIO(data))
 
        st.markdown("---")
        render_api_diagnostics()
        st.caption("Powered by Streamlit, Ultralytics YOLOv8 & Google Gemini")
 
    return input_image, image_sig, lang_level
 
 
# ==========================================
# 6. 各頁籤渲染函式
# ==========================================
def render_tab_quick_record(plotted_rgb, detected_label, best_conf, traffic_light, en_word):
    """⚡ 快速健康紀錄：AI 視覺辨識 + 食科紅綠燈 + 生醫警語"""
    col1, col2 = st.columns(2, gap="large")
 
    # 左：📷 [AI 視覺辨識]
    with col1:
        st.markdown("""
        <div class="card-box">
            <div class="card-header">📷 [AI 視覺辨識]</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(plotted_rgb, caption=f"標籤: {detected_label} | 置信度: {best_conf:.1%}", use_container_width=True)
        st.markdown(
            f"**AI 辨識食物標籤**：`{detected_label}` "
            f"<span class='badge badge-label'>信心度：{best_conf:.1%}</span>",
            unsafe_allow_html=True
        )
 
    # 右：🚦 [食科紅綠燈]
    with col2:
        st.markdown("""
        <div class="card-box">
            <div class="card-header">🚦 [食科紅綠燈]</div>
        </div>
        """, unsafe_allow_html=True)
 
        # 根據燈號給予對應提示色彩
        if "🔴" in str(traffic_light) or "紅燈" in str(traffic_light):
            st.error(f"### {traffic_light}")
            st.markdown("⚠️ **營養評價**：屬於高熱量、高鈉或過度加工類別，建議嚴格控管攝取頻率。")
        elif "🟡" in str(traffic_light) or "黃燈" in str(traffic_light):
            st.warning(f"### {traffic_light}")
            st.markdown("⚖️ **營養評價**：屬於適量食用類別，請搭配不同食物均衡攝取。")
        else:
            st.success(f"### {traffic_light}")
            st.markdown("🌿 **營養評價**：富含優質微量元素或高膳食纖維，屬健康推薦食材！")
 
        st.metric(label="健康等級", value=str(traffic_light).split()[0] + " 評級")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # 下：🩺 [生醫 AI 動態分析]
    st.markdown("""
    <div class="card-box">
        <div class="card-header">🩺 [生醫 AI 動態分析]</div>
    </div>
    """, unsafe_allow_html=True)
    with st.spinner("生醫系 AI 正在為您生成專屬分析..."):
        dynamic_warning = generate_dynamic_warning(en_word)
    st.info(f"**🔬 來自生醫系 AI 的專屬提醒：**\n\n{dynamic_warning}")
    st.caption("※ 本衛教內容由 Gemini AI 生成，僅供日常健康生活管理參考。")
 
 
def render_tab_language(en_word, jp_word, quiz_question, quiz_ans, lang_level):
    """🌍 外語文化探索：雙語單字 + 分級例句(滑鼠懸停翻譯) + 文化小知識"""
    st.markdown("""
    <div class="card-box">
        <div class="card-header">🎌 [外語與文化微學習]</div>
    </div>
    """, unsafe_allow_html=True)
 
    # 雙語單字對照卡片
    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
        st.metric(label="英語 English", value=str(en_word))
    with sub_c2:
        st.metric(label="日語 日本語", value=str(jp_word))
 
    # 🔊 食物單字發音
    pc1, pc2 = st.columns(2)
    with pc1:
        render_speech_panel([(str(en_word), str(en_word))], "en-US", height=50)
    with pc2:
        render_speech_panel([(clean_speech_text(jp_word), clean_speech_text(jp_word))], "ja-JP", height=50)
 
    st.markdown("---")
    st.markdown(f"##### 📖 {lang_level}程度例句　<small>（把滑鼠移到<u>虛線單字</u>上看翻譯）</small>",
                unsafe_allow_html=True)
 
    with st.spinner("AI 語言老師正在撰寫例句..."):
        culture, used_fallback = generate_culture_text(str(en_word), str(jp_word), lang_level)
    if used_fallback:
        st.caption("⚠️ AI 例句暫時無法生成，以下為內建基礎例句。")
 
    lc1, lc2 = st.columns(2, gap="large")
    for col, lang_key, title in ((lc1, "en", "🇬🇧 English"), (lc2, "jp", "🇯🇵 日本語")):
        block = culture[lang_key]
        with col:
            st.markdown(f"**{title}**")
            abbr_html = build_abbr_html(block["sentence"], block["vocab"])
            # ★ 使用 unsafe_allow_html 渲染 <abbr>；內容已在 build_abbr_html 內完成 escape
            st.markdown(f'<div class="abbr-sentence">{abbr_html}</div>', unsafe_allow_html=True)
            st.caption(f"中文：{html.escape(block['translation_zh'])}")
            # 🔊 整句 + 每個單字的發音按鈕
            speech_items = [("整句", block["sentence"])] + [
                (str(v.get("word", "")), str(v.get("word", "")))
                for v in block["vocab"] if isinstance(v, dict) and v.get("word")
            ]
            render_speech_panel(speech_items, "en-US" if lang_key == "en" else "ja-JP", height=90)
            render_tts_audio(block["sentence"], "en" if lang_key == "en" else "ja")
 
    st.caption("💡 手機等觸控裝置無法「懸停」，可長按單字查看；完整單字表如下。")
    with st.expander("📚 本句單字表"):
        for lang_key, title in (("en", "English"), ("jp", "日本語")):
            vocab_df = pd.DataFrame(culture[lang_key]["vocab"])
            if not vocab_df.empty and {"word", "meaning"} <= set(vocab_df.columns):
                st.markdown(f"**{title}**")
                st.dataframe(
                    vocab_df.rename(columns={"word": "單字", "meaning": "意思"}),
                    hide_index=True, use_container_width=True
                )
 
    # 原本的文化小測驗（CSV 題庫）保留為「文化小知識」
    with st.expander("💡 文化小知識：點我展開隨堂提問"):
        st.markdown(f"**題目：{quiz_question}**")
        if st.checkbox("🙋 查看解答", key="reveal_quiz_answer"):
            st.markdown(f"🎉 **正解：** :green[**{quiz_ans}**]")
 
 
def render_tab_quiz(row, food_df, quiz_id, lang_level):
    """🎯 每日測驗任務：根據辨識結果動態出題，含狀態鎖"""
    ss = st.session_state
    ensure_quiz(row, food_df, quiz_id, lang_level)
    quiz = ss.quiz_data
 
    st.markdown("""
    <div class="card-box">
        <div class="card-header">🎯 [每日測驗任務]</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"答對可獲得 +{EXP_PER_QUIZ} EXP｜每張照片只有一次作答機會")
 
    st.markdown(f"#### ❓ {quiz['question']}")
 
    # radio 的 key 含 nonce：每次 reset 後都是全新元件，不會殘留上一題的選取
    st.radio(
        "請選擇答案：",
        options=quiz["options"],
        index=None,
        key=f"quiz_choice_{ss.quiz_nonce}",
        disabled=ss.quiz_answered,           # 【狀態鎖】已作答 -> 選項鎖定
    )
 
    st.button(
        "✅ 送出答案",
        on_click=submit_answer,              # 加分邏輯全在 callback 內
        disabled=ss.quiz_answered,           # 【狀態鎖】已作答 -> 按鈕鎖定，無法連點刷分
        type="primary",
    )
 
    if ss.quiz_warn and not ss.quiz_answered:
        st.warning("請先選擇一個答案再送出喔！")
 
    if ss.quiz_answered:
        if ss.quiz_correct:
            st.success(f"✅ 答對了！ {ss.quiz_reward_msg}")
        else:
            st.error(f"❌ 答錯了～正解是：**{quiz['answer']}**")
        if quiz.get("explanation"):
            st.info(f"📝 解說：{quiz['explanation']}")
        st.caption("想繼續賺 EXP？請在左側換一張新的食物照片！")
 
 
# ==========================================
# 7. 主程序
# ==========================================
def main():
    # 0) 初始化玩家進度與 Streak（必須在 render_sidebar 之前）
    init_session_state()
 
    # 載入模型與衛教資料庫
    model = load_yolo_model("yolov8n.pt")
    food_df = load_food_database("food_database.csv")
 
    # 取得側邊欄影像輸入（含玩家儀表板、語言程度）
    image, image_sig, lang_level = render_sidebar()
 
    # 主畫面標題與導言
    st.markdown('<div class="main-title">🥗 Bite & Learn 跨界識食</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">從餐盤看見全世界：AI 物件辨識 ✕ 營養衛教 ✕ 生醫提醒 ✕ 跨語言文化互動</div>',
        unsafe_allow_html=True
    )
 
    # 未上傳圖片時的引導介面
    if image is None:
        st.info("👈 請從左側側邊欄「上傳食物照片」或「開啟相機拍照」開始體驗！")
 
        # 展開預覽目前資料庫已支援的食物導覽
        with st.expander("📋 點此查看資料庫目前支援的食物品項"):
            st.dataframe(
                food_df[["AI_Label", "EN_Word", "JP_Word", "Traffic_Light"]],
                use_container_width=True,
                hide_index=True
            )
        return
 
    # 確保圖片色彩空間為 RGB
    img_rgb = image.convert("RGB")
 
    # 執行 YOLOv8 物件偵測推論
    with st.spinner("AI 正在仔細辨識您的餐點..."):
        results = model.predict(source=img_rgb, conf=0.25, verbose=False)
 
    first_result = results[0]
    boxes = first_result.boxes
 
    # 【防呆機制 1】畫面中未偵測到任何物件
    if len(boxes) == 0:
        st.warning("⚠️ 畫面中找不到食物，請換張照片試試看喔！")
        st.image(img_rgb, caption="您上傳的原始圖片", width=420)
        return
 
    # 擷取信心度 (confidence) 最高的物件預測結果
    best_idx = int(boxes.conf.argmax())
    best_conf = float(boxes.conf[best_idx].cpu().numpy())
    best_cls_id = int(boxes.cls[best_idx].cpu().numpy())
    detected_label = str(first_result.names[best_cls_id]).strip().lower()
 
    # 繪製辨識框圖片 (results[0].plot() 回傳為 BGR 陣列，需轉為 RGB)
    plotted_bgr = first_result.plot()
    plotted_rgb = plotted_bgr[..., ::-1]
 
    # 資料庫比對
    matched = food_df[food_df["AI_Label"] == detected_label]
 
    # 【防呆機制 2】偵測到的物件不在 CSV 資料庫中
    if matched.empty:
        st.warning(f"ℹ️ AI 認出這是 [{detected_label}]，但目前不在我們的健康資料庫中。")
        st.image(plotted_rgb, caption=f"AI 標註畫面（偵測標籤: {detected_label}，信心度: {best_conf:.1%}）", width=520)
        st.info("💡 提示：本衛教系統主要針對常見食物（如 pizza, apple, banana 等）。歡迎將更多食材加入至 `food_database.csv`！")
        return
 
    # 比對成功，擷取資料欄位
    row = matched.iloc[0]
    en_word = row["EN_Word"]
    jp_word = row["JP_Word"]
    traffic_light = row["Traffic_Light"]
    quiz_question = row["Quiz_Question"]
    quiz_ans = row["Quiz_Ans"]
 
    # 測驗 ID = 圖片簽章 + 食物標籤：同一張圖同一食物 => 同一個 ID (用於防重複領獎與題目快取)
    quiz_id = f"{image_sig}:{detected_label}"
 
    st.success(f"🎯 成功辨識餐點：**{en_word}** (`{detected_label}`)！信心度：**{best_conf:.1%}**")
 
    # ==========================================
    # 三個頁籤
    # ==========================================
    tab_record, tab_lang, tab_quiz = st.tabs(
        ["⚡ 快速健康紀錄", "🌍 外語文化探索", "🎯 每日測驗任務"]
    )
 
    with tab_record:
        render_tab_quick_record(plotted_rgb, detected_label, best_conf, traffic_light, en_word)
 
    with tab_lang:
        render_tab_language(en_word, jp_word, quiz_question, quiz_ans, lang_level)
 
    with tab_quiz:
        render_tab_quiz(row, food_df, quiz_id, lang_level)
 
 
# ==========================================
# 程式進入點
# ==========================================
if __name__ == "__main__":
    main()
