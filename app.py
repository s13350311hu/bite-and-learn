# -*- coding: utf-8 -*-
"""
Bite & Learn 跨界識食 (V3) —— 基於 AI 視覺與大數據之跨國食品翻轉教育與智慧衛教平台
=====================================================================================
目標族群：飲食不健康或不規律的大學生。
一邊管理飲食，一邊學習飲食健康知識與中／日／英三語。

【功能對照 (對應企劃書)】
  1. 飲食紀錄(核心) ........ 「⚡ 快速健康紀錄」：拍照→YOLO 辨識→AI 選項與說明→一鍵記錄、記錄提醒
  2. 知識與語言學習 ........ 「🌍 外語文化探索」「🎯 每日測驗任務」：中日英三語、食物故事、分級測驗、發音
  3. 遊戲化與獎勵 ........... 側邊欄玩家儀表板：EXP / 等級 / 連續登入 / 每日任務 / 徽章
  4. 健康管理延伸 ........... 「💪 健康管理」：代謝計算、體重追蹤、食譜推薦、運動菜單、運動陪伴
  5. 社群 ................... 「👥 社群」：食譜／運動／成果分享，公開或群組可見
  6. 系統 ................... 帳號 + 資料庫 → 跨裝置同步 (登入同一帳號即可)

【技術架構】
  前端：Streamlit ｜ 視覺：Ultralytics YOLOv8 ｜ LLM：Google Gemini
  資料層：SQLAlchemy。預設 SQLite (本機測試用)；在 secrets 設定 DATABASE_URL (Postgres，如 Supabase / Neon)
          即可真正跨裝置、重啟不遺失。

【secrets.toml 需要的設定】
  GEMINI_API_KEY = "..."                      # 必填
  GEMINI_MODEL   = "gemini-2.5-flash"         # 選填：覆蓋預設模型名稱
  DATABASE_URL   = "postgresql://user:pw@host:5432/db"   # 強烈建議：Streamlit Cloud 檔案系統是暫存的

【session_state 鍵值總覽】(維護者請先讀這段)
  --- 登入與通用 ---
  uid              : int|None  目前登入的使用者 id (None = 尚未登入)
  flash            : list      待顯示的提示訊息 [(kind, msg)]，由 show_flash() 統一顯示後清空
  last_touch_day   : str       本 session 最近一次更新「連續登入」的日期
  lang_level       : str       語言程度 (selectbox 的 key；變更時 callback 會存進資料庫)
  llm_errors       : list      最近的 LLM 錯誤，顯示在「API 診斷」
  --- 測驗狀態 (每次換圖 / 換程度 / 換食物都會被 reset_quiz_state() 清空) ---
  quiz_id          : str       目前題目對應的「食物:語言程度」
  quiz_data        : dict      題目 {question, options, answer, explanation, source}
  quiz_answered    : bool      【狀態鎖】True = 已作答，選項與按鈕鎖定
  quiz_correct / quiz_selected / quiz_reward_msg / quiz_warn
  quiz_nonce       : int       radio 元件版本號，改變它 = 全新元件 (清除舊選取)

【重要設計原則】
  * 「分數、EXP、獎勵領取」一律寫進資料庫，而不是只放在 session_state：
    這樣換裝置、重新整理、開多個分頁都不會重置，也無法靠重整刷分。
  * 所有改資料的動作都放在 callback (on_click / on_change) 或 form submit，
    避免 Streamlit 每次 rerun 重複執行造成重複寫入。
"""

import os
import re
import io
import json
import html
import hmac
import hashlib
import random
import secrets as pysecrets
from datetime import datetime, timedelta, timezone, time as dtime

import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
from ultralytics import YOLO
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Float, Text,
    ForeignKey, select, insert, update, delete, func, and_
)
from sqlalchemy.exc import IntegrityError

# gTTS 為選用套件：裝了就多一條可播放音訊 (st.audio)，沒裝則只用瀏覽器內建語音
try:
    from gtts import gTTS
    GTTS_OK = True
except Exception:
    GTTS_OK = False

# ==========================================
# 0. 全域常數
# ==========================================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    pass  # 沒設金鑰時 LLM 呼叫會失敗並走備援內容，原因會顯示在「API 診斷」

try:
    GEMINI_MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-3.8-flash")
except Exception:
    GEMINI_MODEL_NAME = "gemini-3.8-flash"
llm_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

TZ_TAIPEI = timezone(timedelta(hours=8))   # 一律以台北時間計算「今天」

# --- 遊戲化 ---
EXP_PER_QUIZ = 10          # 答對測驗
QUIZ_DAILY_CAP = 5         # 每天最多可領幾次測驗獎勵 (防止拍一堆照片刷分)
EXP_PER_LEVEL = 100
MISSIONS = {               # 每日任務：key -> (名稱, 獎勵 EXP)
    "log_meal": ("記錄 1 餐", 15),
    "story":    ("讀完 1 則食物故事", 10),
    "quiz":     ("答對 1 題測驗", 10),
    "move":     ("完成 1 次運動", 15),
}

# --- 語言程度 ---
LANG_LEVELS = ["零基礎", "基礎", "進階"]
LEVEL_PROMPT_HINT = {
    "零基礎": "學習者完全沒學過該語言。句子必須極短（英文 5~8 個字、日文 10 字內），只用最基礎的單字。",
    "基礎": "學習者具備基礎文法與日常單字。句子 1~2 句，長度適中，可使用簡單的連接詞。",
    "進階": "學習者已有中高級能力。句子可包含較道地的慣用語、複合句或文化典故，詞彙可較豐富。",
}
QUIZ_SPEC = {
    "零基礎": ("題型：單字配對。問『{en} 的中文是什麼』或『日文 {jp} 是哪一種食物』這類最基礎的辨識題。"
              "題幹用繁體中文；干擾選項請用其他常見食物，不可刁鑽。"),
    "基礎": ("題型：日常句型填空或單字應用。例如『I like ___ . (apple)』或『{jp} 要搭配哪個動詞/助詞』。"
            "題幹可含簡單英文或日文短句，並附中文提示；選項為單字或短語。"),
    "進階": ("題型：飲食文化、慣用語、同義詞辨析或營養科學延伸。題幹與選項可以是英文或日文完整句子，"
            "需要推理或文化背景知識才能答對；干擾選項要有迷惑性。"),
}

# --- 飲食紀錄 ---
MEAL_TYPES = ["早餐", "午餐", "晚餐", "點心/宵夜"]
DEFAULT_REMIND = {"enabled": True, "早餐": "08:00", "午餐": "12:30", "晚餐": "18:30"}
KCAL_DEFAULT = {   # 每「一份」的估計熱量 (kcal)，僅供日常管理參考；CSV 若有 Kcal 欄位則以 CSV 為準
    "pizza": 285, "apple": 95, "banana": 105, "orange": 62, "sandwich": 300,
    "broccoli": 55, "hot dog": 290, "donut": 250, "cake": 350, "carrot": 25,
}
ZH_NAMES = {
    "pizza": "披薩", "apple": "蘋果", "banana": "香蕉", "orange": "柳橙", "sandwich": "三明治",
    "broccoli": "青花菜", "hot dog": "熱狗", "donut": "甜甜圈", "cake": "蛋糕", "carrot": "胡蘿蔔",
}
KCAL_FALLBACK = 200.0

# --- 健康管理 ---
ACTIVITY_LEVELS = {
    "久坐（幾乎不運動）": 1.2, "輕度（每週運動 1-3 天）": 1.375,
    "中度（每週運動 3-5 天）": 1.55, "高度（每週運動 6-7 天）": 1.725,
}
GOALS = {"維持體重": 0, "溫和減脂 (-300 kcal)": -300, "增肌／增重 (+300 kcal)": 300}
KCAL_FLOOR = {"男": 1500, "女": 1200, "不指定": 1350}   # 目標熱量下限，避免給出過低的建議
EXERCISES_MET = {   # 運動 -> MET 值 (代謝當量)，熱量 = MET x 體重(kg) x 小時
    "步行／健走": 3.5, "慢跑": 7.0, "騎腳踏車": 6.8, "游泳": 6.0, "重量訓練": 5.0,
    "瑜伽／伸展": 2.5, "跳繩": 11.0, "球類運動": 6.5, "HIIT 間歇訓練": 8.0,
}

# --- 社群 ---
POST_KINDS = ["🍳 食譜", "🏃 運動", "🏆 成果"]
POST_DAILY_CAP = 10

# ==========================================
# 1. 頁面設定與 CSS
# ==========================================
st.set_page_config(page_title="Bite & Learn 跨界識食", page_icon="🥗",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #64748B; margin-bottom: 1.2rem; }
    .card-box { background-color: #FFFFFF; border-radius: 12px; padding: 1.0rem 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #E2E8F0; margin-bottom: 0.8rem; }
    .card-header { font-size: 1.15rem; font-weight: 700; color: #0F172A; display: flex; align-items: center; gap: .5rem; }
    .badge { display: inline-block; padding: .25rem .6rem; border-radius: 9999px; font-size: .85rem; font-weight: 600; }
    .badge-label { background-color: #EEF2FF; color: #4F46E5; }
    .abbr-sentence { font-size: 1.35rem; line-height: 2.1rem; padding: .6rem .2rem; }
    .abbr-sentence abbr { text-decoration: underline dotted #4F46E5; text-underline-offset: 5px;
        cursor: help; color: #4F46E5; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def card_header(title: str):
    """統一的卡片標題 (title 只能是程式內寫死的字串，不可放使用者輸入)"""
    st.markdown(f'<div class="card-box"><div class="card-header">{title}</div></div>', unsafe_allow_html=True)


def md_escape(text) -> str:
    """
    把「使用者輸入 / LLM 輸出」轉成安全的 Markdown 純文字：
    跳脫所有 Markdown 特殊字元，避免對方塞連結、圖片或 :color[...] 之類語法。
    """
    t = re.sub(r"([\\`*_{}\[\]()#+\-.!|>~$<:])", r"\\\1", str(text))
    return t.replace("\n", "  \n")


# ==========================================
# 2. 資料層 (SQLAlchemy；SQLite / Postgres 通用)
# ==========================================
metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("nickname", String(40), unique=True, nullable=False),
    Column("pw_hash", String(128), nullable=False),
    Column("salt", String(64), nullable=False),
    Column("created_at", String(19)),
    Column("exp", Integer, nullable=False, default=0),
    Column("streak", Integer, nullable=False, default=0),
    Column("last_login", String(10)),
    Column("lang_level", String(10), default="零基礎"),
    Column("sex", String(10)), Column("age", Integer), Column("height_cm", Float),
    Column("activity", String(40)), Column("goal", String(40)),
    Column("remind_json", Text),
    Column("fail_count", Integer, nullable=False, default=0),
    Column("locked_until", String(19)),
)
meals = Table(
    "meals", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), index=True, nullable=False),
    Column("day", String(10), index=True), Column("ts", String(19)),
    Column("meal_type", String(10)), Column("label", String(60)), Column("name", String(120)),
    Column("kcal", Float), Column("traffic", String(60)), Column("portion", Float),
    Column("source", String(10)),
    Column("dedupe", String(160), unique=True),     # 同一張照片同一餐不重複記錄
)
weights = Table(
    "weights", metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("day", String(10), primary_key=True), Column("kg", Float, nullable=False),
)
workouts = Table(
    "workouts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), index=True, nullable=False),
    Column("day", String(10), index=True), Column("ts", String(19)),
    Column("kind", String(30)), Column("minutes", Integer), Column("kcal", Float),
)
quiz_rewards = Table(     # 複合主鍵 = 資料庫層級的防刷分鎖
    "quiz_rewards", metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("reward_key", String(120), primary_key=True),
    Column("day", String(10), index=True),
)
mission_claims = Table(
    "mission_claims", metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("day", String(10), primary_key=True), Column("mission", String(20), primary_key=True),
)
badges = Table(
    "badges", metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("badge", String(30), primary_key=True), Column("ts", String(19)),
)
groups_t = Table(
    "groups", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(40), nullable=False), Column("code", String(12), unique=True, nullable=False),
    Column("owner_id", Integer, ForeignKey("users.id"), nullable=False),
)
group_members = Table(
    "group_members", metadata,
    Column("group_id", Integer, ForeignKey("groups.id"), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
)
posts = Table(
    "posts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), index=True, nullable=False),
    Column("ts", String(19)), Column("kind", String(10)),
    Column("title", String(80)), Column("body", Text),
    Column("visibility", String(10)),                        # public / group
    Column("group_id", Integer, ForeignKey("groups.id"), nullable=True),
)
post_likes = Table(
    "post_likes", metadata,
    Column("post_id", Integer, ForeignKey("posts.id"), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
)


def _now() -> datetime:
    return datetime.now(TZ_TAIPEI)


def today_str() -> str:
    return _now().strftime("%Y-%m-%d")


def now_str() -> str:
    return _now().strftime("%Y-%m-%d %H:%M:%S")


@st.cache_resource(show_spinner="連接資料庫中...")
def get_engine():
    """建立資料庫連線 (整個 app 共用一個)，並自動建立不存在的資料表。"""
    url = None
    try:
        url = st.secrets.get("DATABASE_URL")
    except Exception:
        url = None
    url = url or os.environ.get("DATABASE_URL") or "sqlite:///bite_learn.db"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}   # Streamlit 會跨執行緒使用連線
    engine = create_engine(url, **kwargs)
    metadata.create_all(engine)
    return engine


def db_is_local() -> bool:
    return get_engine().dialect.name == "sqlite"


def db_rows(stmt) -> list:
    """執行 SELECT，回傳 list[dict]"""
    with get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(stmt).mappings().all()]


def db_exec(stmt):
    with get_engine().begin() as conn:
        return conn.execute(stmt)


# ---------- 帳號 ----------
def _hash_pw(pw: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), bytes.fromhex(salt_hex), 200_000).hex()


def register_user(nick: str, pw: str):
    """回傳 (uid, 錯誤訊息)"""
    nick = (nick or "").strip()
    if not (2 <= len(nick) <= 20) or not re.fullmatch(r"[\w一-鿿\-]+", nick):
        return None, "暱稱需 2~20 字，只能使用中英文、數字、底線與連字號"
    if len(pw or "") < 6:
        return None, "密碼至少 6 個字元"
    salt = pysecrets.token_hex(16)
    try:
        res = db_exec(insert(users).values(
            nickname=nick, pw_hash=_hash_pw(pw, salt), salt=salt, created_at=now_str(),
            exp=0, streak=0, lang_level="零基礎", fail_count=0,
            remind_json=json.dumps(DEFAULT_REMIND)))
        return res.inserted_primary_key[0], None
    except IntegrityError:
        return None, "這個暱稱已經有人使用了"


def login_user(nick: str, pw: str):
    """回傳 (uid, 錯誤訊息)。連續輸錯 5 次鎖定 5 分鐘，防止暴力猜密碼。"""
    rows = db_rows(select(users).where(users.c.nickname == (nick or "").strip()))
    if not rows:
        return None, "暱稱或密碼錯誤"
    u = rows[0]
    if u["locked_until"] and u["locked_until"] > now_str():
        return None, f"帳號暫時鎖定，請在 {u['locked_until'][11:16]} 之後再試"
    if hmac.compare_digest(_hash_pw(pw or "", u["salt"]), u["pw_hash"]):
        db_exec(update(users).where(users.c.id == u["id"]).values(fail_count=0, locked_until=None))
        return u["id"], None
    fails = (u["fail_count"] or 0) + 1
    if fails >= 5:
        until = (_now() + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        db_exec(update(users).where(users.c.id == u["id"]).values(fail_count=0, locked_until=until))
        return None, "連續錯誤 5 次，帳號暫時鎖定 5 分鐘"
    db_exec(update(users).where(users.c.id == u["id"]).values(fail_count=fails))
    return None, "暱稱或密碼錯誤"


def get_user(uid):
    rows = db_rows(select(users).where(users.c.id == uid))
    return rows[0] if rows else None


def touch_login(uid: int):
    """更新連續登入天數 (Streak)：昨天登入過 +1，今天已登入就不動，否則重設為 1。"""
    u = get_user(uid)
    today = today_str()
    if not u or u["last_login"] == today:
        return
    yesterday = (_now() - timedelta(days=1)).strftime("%Y-%m-%d")
    streak = (u["streak"] or 0) + 1 if u["last_login"] == yesterday else 1
    db_exec(update(users).where(users.c.id == uid).values(streak=streak, last_login=today))


def change_password(uid: int, old: str, new: str):
    u = get_user(uid)
    if not u or not hmac.compare_digest(_hash_pw(old or "", u["salt"]), u["pw_hash"]):
        return "目前的密碼不正確"
    if len(new or "") < 6:
        return "新密碼至少 6 個字元"
    salt = pysecrets.token_hex(16)
    db_exec(update(users).where(users.c.id == uid).values(pw_hash=_hash_pw(new, salt), salt=salt))
    return None


def delete_account(uid: int):
    """永久刪除帳號與所有相關資料 (含自己建立的群組與群組貼文)"""
    with get_engine().begin() as conn:
        owned = [r[0] for r in conn.execute(select(groups_t.c.id).where(groups_t.c.owner_id == uid))]
        if owned:
            gp = [r[0] for r in conn.execute(select(posts.c.id).where(posts.c.group_id.in_(owned)))]
            if gp:
                conn.execute(delete(post_likes).where(post_likes.c.post_id.in_(gp)))
            conn.execute(delete(posts).where(posts.c.group_id.in_(owned)))
            conn.execute(delete(group_members).where(group_members.c.group_id.in_(owned)))
            conn.execute(delete(groups_t).where(groups_t.c.id.in_(owned)))
        mine = [r[0] for r in conn.execute(select(posts.c.id).where(posts.c.user_id == uid))]
        if mine:
            conn.execute(delete(post_likes).where(post_likes.c.post_id.in_(mine)))
        conn.execute(delete(post_likes).where(post_likes.c.user_id == uid))
        conn.execute(delete(posts).where(posts.c.user_id == uid))
        conn.execute(delete(group_members).where(group_members.c.user_id == uid))
        for t in (meals, weights, workouts, quiz_rewards, mission_claims, badges):
            conn.execute(delete(t).where(t.c.user_id == uid))
        conn.execute(delete(users).where(users.c.id == uid))


# ---------- 飲食紀錄 ----------
def add_meal(p: dict) -> bool:
    """新增一筆飲食紀錄；dedupe 衝突 (同照片同餐別重複送出) 回傳 False"""
    try:
        db_exec(insert(meals).values(
            user_id=p["uid"], day=p["day"], ts=now_str(), meal_type=p["meal_type"], label=p["label"],
            name=p["name"], kcal=float(p["kcal"]), traffic=p["traffic"], portion=float(p["portion"]),
            source=p["source"], dedupe=p.get("dedupe")))
        return True
    except IntegrityError:
        return False


def meals_between(uid: int, d1: str, d2: str) -> pd.DataFrame:
    rows = db_rows(select(meals).where(and_(meals.c.user_id == uid, meals.c.day >= d1, meals.c.day <= d2))
                   .order_by(meals.c.ts))
    cols = ["id", "user_id", "day", "ts", "meal_type", "label", "name", "kcal", "traffic", "portion", "source", "dedupe"]
    return pd.DataFrame(rows, columns=cols)


def delete_meal(uid: int, meal_id: int):
    db_exec(delete(meals).where(and_(meals.c.id == meal_id, meals.c.user_id == uid)))


# ---------- 體重 / 運動 ----------
def save_weight(uid: int, kg: float):
    day = today_str()
    with get_engine().begin() as conn:
        conn.execute(delete(weights).where(and_(weights.c.user_id == uid, weights.c.day == day)))
        conn.execute(insert(weights).values(user_id=uid, day=day, kg=float(kg)))


def weight_history(uid: int) -> pd.DataFrame:
    rows = db_rows(select(weights.c.day, weights.c.kg).where(weights.c.user_id == uid).order_by(weights.c.day))
    return pd.DataFrame(rows, columns=["day", "kg"])


def latest_weight(uid: int):
    h = weight_history(uid)
    return float(h["kg"].iloc[-1]) if not h.empty else None


def add_workout(uid: int, kind: str, minutes: int, kcal: float):
    db_exec(insert(workouts).values(user_id=uid, day=today_str(), ts=now_str(), kind=kind,
                                    minutes=int(minutes), kcal=float(kcal)))


def workouts_since(uid: int, d1: str) -> pd.DataFrame:
    rows = db_rows(select(workouts).where(and_(workouts.c.user_id == uid, workouts.c.day >= d1)).order_by(workouts.c.ts))
    return pd.DataFrame(rows, columns=["id", "user_id", "day", "ts", "kind", "minutes", "kcal"])


# ---------- 遊戲化 (EXP / 任務 / 徽章) ----------
def push_flash(kind: str, msg: str):
    """把提示訊息排進佇列，之後由 show_flash() 在主畫面統一顯示"""
    st.session_state.setdefault("flash", []).append((kind, msg))


def claim_quiz_reward(uid: int, reward_key: str) -> str:
    """
    領取測驗獎勵，回傳 'ok' / 'dup' (已領過) / 'cap' (今日上限)。
    reward_key 是資料庫複合主鍵的一部分 -> 即使多分頁、重整、狂點按鈕，同一題也只會成功一次。
    """
    day = today_str()
    try:
        with get_engine().begin() as conn:
            cnt = conn.execute(select(func.count()).select_from(quiz_rewards).where(
                and_(quiz_rewards.c.user_id == uid, quiz_rewards.c.day == day))).scalar()
            if cnt >= QUIZ_DAILY_CAP:
                return "cap"
            conn.execute(insert(quiz_rewards).values(user_id=uid, reward_key=reward_key, day=day))
            conn.execute(update(users).where(users.c.id == uid).values(exp=users.c.exp + EXP_PER_QUIZ))
        return "ok"
    except IntegrityError:
        return "dup"


def missions_done_today(uid: int) -> set:
    rows = db_rows(select(mission_claims.c.mission).where(
        and_(mission_claims.c.user_id == uid, mission_claims.c.day == today_str())))
    return {r["mission"] for r in rows}


def complete_mission(uid: int, mission: str) -> bool:
    """完成今日任務並發放 EXP。同一任務一天只會成功一次 (複合主鍵保證)。"""
    name, exp = MISSIONS[mission]
    try:
        with get_engine().begin() as conn:
            conn.execute(insert(mission_claims).values(user_id=uid, day=today_str(), mission=mission))
            conn.execute(update(users).where(users.c.id == uid).values(exp=users.c.exp + exp))
        push_flash("success", f"🎯 每日任務完成：{name}　+{exp} EXP")
        return True
    except IntegrityError:
        return False


BADGES = {   # 徽章 key -> (名稱, 說明, 條件函式(stats))
    "first_log": ("🍽️ 初次記錄", "記錄第一餐", lambda s: s["meals"] >= 1),
    "log_20":    ("📒 飲食達人", "累積記錄 20 餐", lambda s: s["meals"] >= 20),
    "streak_3":  ("🔥 三日連勝", "連續登入 3 天", lambda s: s["streak"] >= 3),
    "streak_7":  ("🔥 七日連勝", "連續登入 7 天", lambda s: s["streak"] >= 7),
    "quiz_5":    ("🎓 小小學霸", "累積答對 5 題", lambda s: s["quiz"] >= 5),
    "move_3":    ("💪 運動夥伴", "累積運動 3 次", lambda s: s["workouts"] >= 3),
    "sharer":    ("👥 分享達人", "發表第一篇社群貼文", lambda s: s["posts"] >= 1),
    "lv_5":      ("⭐ Lv.5 達成", "等級達到 5", lambda s: s["exp"] // EXP_PER_LEVEL + 1 >= 5),
}


def _count(conn, table, uid) -> int:
    return conn.execute(select(func.count()).select_from(table).where(table.c.user_id == uid)).scalar() or 0


def check_badges(uid: int):
    """依目前統計補發尚未取得的徽章"""
    u = get_user(uid)
    if not u:
        return
    with get_engine().connect() as conn:
        stats = {"meals": _count(conn, meals, uid), "quiz": _count(conn, quiz_rewards, uid),
                 "workouts": _count(conn, workouts, uid), "posts": _count(conn, posts, uid),
                 "streak": u["streak"] or 0, "exp": u["exp"] or 0}
    have = {r["badge"] for r in db_rows(select(badges.c.badge).where(badges.c.user_id == uid))}
    for key, (name, desc, cond) in BADGES.items():
        if key not in have and cond(stats):
            try:
                db_exec(insert(badges).values(user_id=uid, badge=key, ts=now_str()))
                push_flash("success", f"🏅 獲得徽章：{name}（{desc}）")
            except IntegrityError:
                pass


def user_badges(uid: int) -> list:
    return [r["badge"] for r in db_rows(select(badges.c.badge).where(badges.c.user_id == uid))]


# ---------- 社群 ----------
def create_group(uid: int, name: str):
    name = (name or "").strip()
    if not (2 <= len(name) <= 30):
        return None, "群組名稱需 2~30 字"
    for _ in range(5):
        code = "".join(pysecrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        try:
            with get_engine().begin() as conn:
                gid = conn.execute(insert(groups_t).values(name=name, code=code, owner_id=uid)).inserted_primary_key[0]
                conn.execute(insert(group_members).values(group_id=gid, user_id=uid))
            return code, None
        except IntegrityError:
            continue
    return None, "建立失敗，請再試一次"


def join_group(uid: int, code: str):
    rows = db_rows(select(groups_t).where(groups_t.c.code == (code or "").strip().upper()))
    if not rows:
        return "找不到這個邀請碼"
    try:
        db_exec(insert(group_members).values(group_id=rows[0]["id"], user_id=uid))
        return None
    except IntegrityError:
        return "你已經在這個群組裡了"


def my_groups(uid: int) -> list:
    j = group_members.join(groups_t, group_members.c.group_id == groups_t.c.id)
    rows = db_rows(select(groups_t.c.id, groups_t.c.name, groups_t.c.code, groups_t.c.owner_id)
                   .select_from(j).where(group_members.c.user_id == uid).order_by(groups_t.c.id))
    for r in rows:
        r["members"] = db_rows(select(func.count().label("n")).select_from(group_members)
                               .where(group_members.c.group_id == r["id"]))[0]["n"]
    return rows


def create_post(uid: int, kind: str, title: str, body: str, visibility: str, group_id):
    title, body = (title or "").strip(), (body or "").strip()
    if not title or not body:
        return "標題與內容都要填寫"
    if len(title) > 60 or len(body) > 2000:
        return "標題最多 60 字、內容最多 2000 字"
    today_posts = db_rows(select(func.count().label("n")).select_from(posts).where(
        and_(posts.c.user_id == uid, posts.c.ts >= today_str() + " 00:00:00")))[0]["n"]
    if today_posts >= POST_DAILY_CAP:
        return f"今天已達發文上限 ({POST_DAILY_CAP} 篇)"
    if visibility == "group":
        if group_id is None or group_id not in [g["id"] for g in my_groups(uid)]:
            return "請選擇你已加入的群組"
    else:
        group_id = None
    db_exec(insert(posts).values(user_id=uid, ts=now_str(), kind=kind, title=title, body=body,
                                 visibility=visibility, group_id=group_id))
    return None


def fetch_posts(scope: str, uid: int, limit: int = 30) -> list:
    j = posts.join(users, posts.c.user_id == users.c.id).outerjoin(groups_t, posts.c.group_id == groups_t.c.id)
    stmt = select(posts.c.id, posts.c.kind, posts.c.title, posts.c.body, posts.c.ts, posts.c.visibility,
                  posts.c.user_id, users.c.nickname, groups_t.c.name.label("group_name")).select_from(j)
    if scope == "public":
        stmt = stmt.where(posts.c.visibility == "public")
    elif scope == "groups":
        gids = [g["id"] for g in my_groups(uid)]
        if not gids:
            return []
        stmt = stmt.where(and_(posts.c.visibility == "group", posts.c.group_id.in_(gids)))
    else:
        stmt = stmt.where(posts.c.user_id == uid)
    rows = db_rows(stmt.order_by(posts.c.id.desc()).limit(limit))
    if rows:
        ids = [r["id"] for r in rows]
        likes = {r["post_id"]: r["n"] for r in db_rows(
            select(post_likes.c.post_id, func.count().label("n")).where(post_likes.c.post_id.in_(ids))
            .group_by(post_likes.c.post_id))}
        mine = {r["post_id"] for r in db_rows(
            select(post_likes.c.post_id).where(and_(post_likes.c.user_id == uid, post_likes.c.post_id.in_(ids))))}
        for r in rows:
            r["likes"], r["liked"] = likes.get(r["id"], 0), r["id"] in mine
    return rows


def toggle_like(uid: int, post_id: int):
    exists = db_rows(select(post_likes.c.post_id).where(and_(post_likes.c.post_id == post_id, post_likes.c.user_id == uid)))
    if exists:
        db_exec(delete(post_likes).where(and_(post_likes.c.post_id == post_id, post_likes.c.user_id == uid)))
    else:
        try:
            db_exec(insert(post_likes).values(post_id=post_id, user_id=uid))
        except IntegrityError:
            pass


def delete_post(uid: int, post_id: int):
    with get_engine().begin() as conn:
        owned = conn.execute(select(posts.c.id).where(and_(posts.c.id == post_id, posts.c.user_id == uid))).first()
        if owned:
            conn.execute(delete(post_likes).where(post_likes.c.post_id == post_id))
            conn.execute(delete(posts).where(posts.c.id == post_id))


# ==========================================
# 3. 核心資源：YOLO、食物資料庫、辨識
# ==========================================
@st.cache_resource(show_spinner="正在載入 YOLOv8 視覺模型中...")
def load_yolo_model(model_name: str = "yolov8n.pt") -> YOLO:
    """載入 YOLOv8 預訓練模型；cache_resource 避免重複載入權重。"""
    return YOLO(model_name)


@st.cache_data(show_spinner="載入食物衛教資料庫...")
def load_food_database(csv_path: str = "food_database.csv") -> pd.DataFrame:
    """
    讀取 food_database.csv；檔案不存在時自動產生示範資料庫。
    V3 起會自動補上 ZH_Word (中文名) 與 Kcal (每份熱量) 欄位，舊版 CSV 不用改也能跑。
    """
    if not os.path.exists(csv_path):
        sample_data = {
            "AI_Label": ["pizza", "apple", "banana", "orange", "sandwich",
                         "broccoli", "hot dog", "donut", "cake", "carrot"],
            "EN_Word": ["Pizza", "Apple", "Banana", "Orange", "Sandwich",
                        "Broccoli", "Hot Dog", "Donut", "Cake", "Carrot"],
            "JP_Word": ["ピザ (Piza)", "りんご (Ringo)", "バナナ (Banana)", "オレンジ (Orenji)", "サンドイッチ (Sandoitchi)",
                        "ブロッコリー (Burokkorī)", "ホットドッグ (Hottodoggu)", "ドーナツ (Dōnatsu)", "ケーキ (Kēki)", "にんじん (Ninjin)"],
            "Traffic_Light": ["🔴 紅燈 (高鈉/高脂)", "🟢 綠燈 (高纖維)", "🟡 黃燈 (高鉀/天然糖份)", "🟢 綠燈 (高維他命C)",
                              "🟡 黃燈 (碳水均衡/視夾餡)", "🟢 綠燈 (超抗氧化)", "🔴 紅燈 (高鈉/超加工肉品)",
                              "🔴 紅燈 (超精製糖/反式脂肪)", "🔴 紅燈 (精製糖/高熱量)", "🟢 綠燈 (富含β-胡蘿蔔素)"],
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
                "脂溶性維生素的最佳來源，建議搭配健康油脂烹調以促進吸收，有助眼睛保健。"],
            "Quiz_Question": [
                "在義大利，披薩上加什麼會觸怒當地人？", "日本最有名的蘋果產地在哪個縣？",
                "香蕉彎曲生長的主要原因是什麼？", "橙色柑橘中主要提供亮麗橘色的抗氧化色素是什麼？",
                "三明治相傳是由哪一國的「三明治伯爵」所發明的？", "青花菜 (Broccoli) 與花椰菜 (Cauliflower) 其實是同一物種嗎？",
                "熱狗的名字由來傳說與哪一種狗的外型有關？", "為什麼傳統甜甜圈中間通常會有一個洞？",
                "法式經典「瑪德蓮蛋糕」著名的貝殼外型是由哪種容器烤出來的？", "兔子在自然環境中最常吃的食物其實是胡蘿蔔嗎？"],
            "Quiz_Ans": [
                "鳳梨 (夏威夷披薩)", "青森縣", "背地性（向陽性，朝陽光方向生長）", "胡蘿蔔素與類黃酮",
                "英國 (John Montagu, 4th Earl of Sandwich)", "是的，兩者都是甘藍的變種", "臘腸犬 (Dachshund)",
                "為了讓油炸時受熱均勻，避免中心炸不熟", "貝殼形模具",
                "不是，野兔主要吃草和葉子，胡蘿蔔糖分過高反而不能多吃"],
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

    df["AI_Label"] = df["AI_Label"].astype(str).str.strip().str.lower()
    if "ZH_Word" not in df.columns:
        df["ZH_Word"] = df["AI_Label"].map(ZH_NAMES).fillna(df["EN_Word"])
    if "Kcal" not in df.columns:
        df["Kcal"] = df["AI_Label"].map(KCAL_DEFAULT)
    df["Kcal"] = pd.to_numeric(df["Kcal"], errors="coerce").fillna(KCAL_FALLBACK)
    return df


def traffic_emoji(traffic) -> str:
    """從 '🔴 紅燈 (...)' 取出燈號 emoji；沒有則視為黃燈"""
    s = str(traffic)
    for e in ("🔴", "🟡", "🟢"):
        if e in s:
            return e
    return "🟡"


def food_row(food_df: pd.DataFrame, label: str):
    m = food_df[food_df["AI_Label"] == label]
    return m.iloc[0] if not m.empty else None


@st.cache_data(show_spinner=False, max_entries=16)
def run_detection(img_bytes: bytes) -> dict:
    """
    YOLO 推論 (以圖片內容快取)：Streamlit 每次互動都會 rerun，
    若不快取，使用者每點一個按鈕都會重新跑一次模型。
    回傳 {plotted: 標註後 RGB 圖, dets: [(label, conf)...]}
    """
    model = load_yolo_model("yolov8n.pt")
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    res = model.predict(source=img, conf=0.25, verbose=False)[0]
    dets = []
    if len(res.boxes) > 0:
        for c, k in zip(res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            dets.append((str(res.names[int(k)]).strip().lower(), float(c)))
        plotted = res.plot()[..., ::-1]          # BGR -> RGB
    else:
        plotted = np.array(img)
    return {"plotted": np.ascontiguousarray(plotted), "dets": dets}


def build_candidates(dets: list, food_df: pd.DataFrame):
    """把偵測結果整理成 (資料庫有的候選 [(label, conf)], 資料庫沒有的 [label])，依信心度排序"""
    best = {}
    for label, conf in dets:
        best[label] = max(best.get(label, 0.0), conf)
    known_labels = set(food_df["AI_Label"])
    ordered = sorted(best.items(), key=lambda x: -x[1])
    return [(l, c) for l, c in ordered if l in known_labels], [l for l, _ in ordered if l not in known_labels]


# ==========================================
# 4. Gemini 呼叫層 (快取 + 容錯 + 錯誤可見)
# ==========================================
@st.cache_data(show_spinner=False, ttl=3600)
def _gemini_generate(prompt: str, json_mode: bool = False) -> str:
    """
    最底層的 Gemini 呼叫。★ 不在這裡捕捉例外：st.cache_data 只快取「成功」的結果，
    若把錯誤訊息當字串 return，429 之類的暫時性錯誤會被快取，之後 API 恢復也一直看到錯誤。
    """
    cfg = {"response_mime_type": "application/json"} if json_mode else None
    return llm_model.generate_content(prompt, generation_config=cfg).text


def _friendly_error(e: Exception) -> str:
    msg = str(e)
    if "429" in msg:
        return "⚠️ 哎呀！大家太熱情了，AI 護理師有點喘不過氣，請等待 10 秒後再試一次喔！"
    return f"⚠️ AI 暫時無法回應（{type(e).__name__}）。可到側邊欄「API 診斷」查看原因。"


def _log_llm_error(tag: str, e: Exception):
    errs = st.session_state.setdefault("llm_errors", [])
    errs.append(f"[{_now():%H:%M:%S}] {tag} -> {type(e).__name__}: {e}")
    del errs[:-6]


def _safe_generate(prompt: str, tag: str, json_mode: bool = False):
    """呼叫 Gemini，回傳 (文字, 例外)。JSON 模式失敗會退回一般模式再試一次。"""
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


def _parse_json(text: str):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned)


def generate_dynamic_warning(food_name: str) -> str:
    prompt = f"""
    你現在是一位充滿幽默感、具備醫學與營養學知識的生醫系大學生。
    使用者剛剛用系統掃描到準備吃「{food_name}」。
    請用繁體中文，用大約 50 到 80 字的一小段話，給予健康警告或營養提示。
    語氣要生動活潑、有點像在吐槽或關心朋友，讓大學生看了會有共鳴。
    """
    text, err = _safe_generate(prompt, "生醫警語")
    return text if text else _friendly_error(err)


# ---------- 外語例句 ----------
def clean_speech_text(text: str) -> str:
    """去掉『ピザ (Piza)』這類括號羅馬拼音，只留要念/要比對的文字"""
    return re.sub(r"\s*[\(（].*?[\)）]", "", str(text)).strip()


def _fallback_culture(en_word: str, jp_word: str) -> dict:
    jp_plain = clean_speech_text(jp_word)
    return {
        "en": {"sentence": f"I want to eat {en_word} today.", "translation_zh": f"我今天想吃 {en_word}。",
               "vocab": [{"word": "eat", "meaning": "吃"}, {"word": "today", "meaning": "今天"},
                         {"word": en_word, "meaning": "（本次辨識的食物）"}]},
        "jp": {"sentence": f"今日は{jp_plain}を食べたいです。", "translation_zh": f"我今天想吃{jp_plain}。",
               "vocab": [{"word": "今日", "meaning": "今天 (きょう)"}, {"word": "食べたい", "meaning": "想吃 (たべたい)"},
                         {"word": jp_plain, "meaning": "（本次辨識的食物）"}]},
    }


def _valid_lang_block(block, key="sentence") -> bool:
    try:
        return (isinstance(block[key], str) and block[key].strip() != ""
                and isinstance(block["vocab"], list))
    except Exception:
        return False


def generate_culture_text(en_word: str, jp_word: str, level: str) -> tuple:
    """依語言程度生成英/日文例句 + 中文翻譯 + 單字表。回傳 (資料 dict, 是否使用備援)"""
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
    raw, _ = _safe_generate(prompt, "外語例句", json_mode=True)
    if raw:
        try:
            data = _parse_json(raw)
            if all(_valid_lang_block(data[k]) and isinstance(data[k].get("translation_zh", ""), str) for k in ("en", "jp")):
                return data, False
            _log_llm_error("外語例句", ValueError("JSON 結構不符預期"))
        except Exception as e:
            _log_llm_error("外語例句(解析)", e)
    return _fallback_culture(en_word, jp_word), True


# ---------- 食物故事 (中日英三語) ----------
def _fallback_story(row: pd.Series, culture_fb: dict) -> dict:
    return {
        "zh": {"title": f"{row['ZH_Word']}的小故事", "text": f"{row['Quiz_Question']}　答案：{row['Quiz_Ans']}"},
        "en": {"text": culture_fb["en"]["sentence"], "vocab": culture_fb["en"]["vocab"]},
        "ja": {"text": culture_fb["jp"]["sentence"], "vocab": culture_fb["jp"]["vocab"]},
        "fun_fact": str(row["Bio_Warning"]),
    }


def generate_story(row: pd.Series, level: str) -> tuple:
    """生成「食物背後的故事、知識與文化」中日英三語版本。回傳 (資料, 是否備援)"""
    zh, en, jp = row["ZH_Word"], row["EN_Word"], clean_speech_text(row["JP_Word"])
    prompt = f"""
    你是一位美食文化作家兼外語老師。請介紹「{zh}（英文 {en}／日文 {jp}）」背後的故事、營養知識或飲食文化。
    學習者程度：{level}。{LEVEL_PROMPT_HINT[level]}
    只回傳 JSON：
    {{"zh": {{"title": "標題(12字內)", "text": "繁體中文故事 80~120 字"}},
      "en": {{"text": "上述內容的英文版(依程度調整長度與用字)", "vocab": [{{"word": "...", "meaning": "繁中意思"}}]}},
      "ja": {{"text": "上述內容的日文版(依程度調整長度與用字)", "vocab": [{{"word": "...", "meaning": "繁中意思"}}]}},
      "fun_fact": "30 字內的繁體中文冷知識"}}
    ★ vocab 各 3~6 個，word 必須逐字出現在對應的 text 裡。
    """
    raw, _ = _safe_generate(prompt, "食物故事", json_mode=True)
    if raw:
        try:
            d = _parse_json(raw)
            if (isinstance(d["zh"]["text"], str) and d["zh"]["text"].strip()
                    and _valid_lang_block(d["en"], "text") and _valid_lang_block(d["ja"], "text")):
                d.setdefault("fun_fact", "")
                d["zh"].setdefault("title", f"{zh}的小故事")
                return d, False
            _log_llm_error("食物故事", ValueError("JSON 結構不符預期"))
        except Exception as e:
            _log_llm_error("食物故事(解析)", e)
    return _fallback_story(row, _fallback_culture(en, jp)), True


def build_abbr_html(sentence: str, vocab: list) -> str:
    """
    把句子中的單字包成 <abbr title="翻譯">單字</abbr>，滑鼠懸停即顯示翻譯。
    安全設計 (後面會用 unsafe_allow_html 渲染 LLM 輸出)：
      1. 先對整句 html.escape，LLM 夾帶的標籤會變成純文字
      2. 用「單次 re.sub + 合併 pattern」替換，避免後面的單字比對到前面 <abbr title> 屬性內的文字
    """
    mapping = {}
    for item in vocab:
        if isinstance(item, dict):
            word, meaning = str(item.get("word", "")).strip(), str(item.get("meaning", "")).strip()
            if word and meaning:
                mapping[word.lower()] = meaning
    escaped_sentence = html.escape(sentence)
    if not mapping:
        return escaped_sentence
    words = sorted(mapping.keys(), key=len, reverse=True)       # 長字優先 (hot dog 先於 hot)
    pattern = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(html.escape(w)) for w in words) + r")(?![A-Za-z])",
                         flags=re.IGNORECASE)

    def _wrap(m):
        original = m.group(1)
        meaning = html.escape(mapping.get(html.unescape(original).lower(), ""), quote=True)
        return f'<abbr title="{meaning}">{original}</abbr>' if meaning else original

    return pattern.sub(_wrap, escaped_sentence)


# ---------- 發音 (TTS) ----------
@st.cache_data(show_spinner=False, ttl=86400)
def tts_audio_bytes(text: str, lang: str) -> bytes:
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    return buf.getvalue()


def render_speech_panel(items: list, lang_code: str, height: int = 70):
    """
    瀏覽器內建 Web Speech API 的發音按鈕列 (免 API、免安裝)。
    items: [(按鈕文字, 要念的文字)]；lang_code: 'en-US' / 'ja-JP' / 'zh-TW'
    要念的文字放在 data-t 屬性 (已 HTML 跳脫)，避免單引號/雙引號破壞語法。
    """
    buttons = "".join(
        f'<button data-t="{html.escape(str(txt), quote=True)}" onclick="speak(this.dataset.t)">🔊 {html.escape(str(label))}</button>'
        for label, txt in items if txt)
    page = """
    <style>
      button {margin:2px 4px 2px 0;padding:4px 10px;border:1px solid #CBD5E1;border-radius:9999px;
              background:#EEF2FF;color:#4F46E5;font-size:14px;cursor:pointer;}
      button:hover {background:#E0E7FF;}
    </style>
    <div>__BUTTONS__</div>
    <script>
      function speak(t) {
        if (!('speechSynthesis' in window)) { alert('此瀏覽器不支援語音合成'); return; }
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(t);
        u.lang = '__LANG__'; u.rate = 0.9;
        window.speechSynthesis.speak(u);
      }
    </script>
    """.replace("__BUTTONS__", buttons).replace("__LANG__", lang_code)
    components.html(page, height=height, scrolling=True)


def render_tts_audio(text: str, gtts_lang: str):
    if not GTTS_OK or not text:
        return
    try:
        st.audio(tts_audio_bytes(text, gtts_lang), format="audio/mp3")
    except Exception as e:
        _log_llm_error("gTTS", e)


# ---------- 動態測驗 ----------
def _fallback_quiz(row: pd.Series, food_df: pd.DataFrame, level: str, seed: str) -> dict:
    """LLM 失敗時的備援，三個程度題型不同：零基礎=看日文選英文；基礎=看英文選日文；進階=CSV 文化題"""
    rng = random.Random(seed)
    en, jp = str(row["EN_Word"]), clean_speech_text(row["JP_Word"])
    others = food_df[food_df["EN_Word"] != row["EN_Word"]]

    def _build(question, correct, pool, explanation):
        pool = [x for x in dict.fromkeys(map(str, pool)) if x != correct]
        options = rng.sample(pool, k=min(3, len(pool))) + [correct]
        rng.shuffle(options)
        return {"question": question, "options": options, "answer": correct,
                "explanation": explanation, "source": "fallback"}

    if level == "零基礎":
        return _build(f"日文的「{jp}」是下列哪一種食物？（選出英文單字）", en,
                      others["EN_Word"].tolist(), f"{jp} 的英文是 {en}。")
    if level == "基礎":
        return _build(f"英文「{en}」的日文怎麼說？", jp,
                      [clean_speech_text(x) for x in others["JP_Word"]], f"{en} 的日文是 {jp}。")
    return _build(str(row["Quiz_Question"]), str(row["Quiz_Ans"]), others["Quiz_Ans"].tolist(),
                  "（本題來自內建文化題庫）")


def generate_quiz(row: pd.Series, food_df: pd.DataFrame, level: str, seed: str) -> dict:
    """依辨識的食物與語言程度，請 Gemini 動態出 1 道四選一選擇題；失敗則用備援題。"""
    en_word, jp_word = row["EN_Word"], clean_speech_text(row["JP_Word"])
    spec = QUIZ_SPEC[level].format(en=en_word, jp=jp_word)
    prompt = f"""
    你是一位外語與飲食文化的出題老師。請針對食物「{en_word}」（日文：{jp_word}）出 1 道四選一選擇題。
    學習者程度：【{level}】。{LEVEL_PROMPT_HINT[level]}
    {spec}
    ★ 題目難度與題型必須明顯符合「{level}」，不可出成其他程度的題目。
    只回傳 JSON：
    {{"question": "...", "options": ["A選項", "B選項", "C選項", "D選項"], "answer": "必須與 options 其中一項完全相同", "explanation": "30 字內的繁體中文解說"}}
    """
    raw, _ = _safe_generate(prompt, "動態測驗", json_mode=True)
    try:
        data = _parse_json(raw)
        options = [str(o).strip() for o in data["options"]]
        answer = str(data["answer"]).strip()
        if len(options) == 4 and len(set(options)) == 4 and answer in options:
            random.Random(seed).shuffle(options)       # 以 seed 洗牌：同一題選項順序固定
            return {"question": str(data["question"]).strip(), "options": options, "answer": answer,
                    "explanation": str(data.get("explanation", "")).strip(), "source": "ai"}
    except Exception as e:
        if raw:
            _log_llm_error("動態測驗(解析)", e)
    return _fallback_quiz(row, food_df, level, seed)


# ---------- 食譜 / 運動菜單 ----------
def generate_recipes(remaining_kcal, goal, meal, restriction, minutes, equipment, variant) -> tuple:
    kcal_txt = f"這一餐熱量請控制在約 {int(remaining_kcal)} kcal 以內" if remaining_kcal else "請以均衡、適量為原則"
    prompt = f"""
    你是一位營養師，服務對象是飲食不規律的大學生。請推薦 2 道{meal}食譜。
    條件：{kcal_txt}；目標：{goal}；飲食限制：{restriction or '無'}；
    可用烹調時間：{minutes} 分鐘；設備：{equipment}；食材要便宜、容易在超商或傳統市場買到。
    每道食譜請用繁體中文 Markdown 輸出：名稱、估計熱量、食材清單、步驟(3~6 步)、一句營養小提醒。
    不要使用表格，不要宣稱療效。（版本 {variant}）
    """
    text, err = _safe_generate(prompt, "食譜推薦")
    return (text, None) if text else (None, _friendly_error(err))


def generate_workout(goal, minutes, equipment, intensity, variant) -> tuple:
    prompt = f"""
    你是一位溫和的運動教練，服務對象是缺乏運動習慣的大學生。請設計一份 {minutes} 分鐘的運動菜單。
    目標：{goal}；器材：{equipment}；強度偏好：{intensity}。
    用繁體中文 Markdown 輸出：暖身、主訓練(列出動作、組數/時間、休息)、收操伸展；
    最後附一句鼓勵與「若感到不適請立即停止」的安全提醒。不要使用表格。（版本 {variant}）
    """
    text, err = _safe_generate(prompt, "運動菜單")
    return (text, None) if text else (None, _friendly_error(err))


# ==========================================
# 5. 狀態管理 (Session State & Callbacks)
# ==========================================
def init_session_state():
    """只在鍵不存在時設定預設值 (rerun 不會覆蓋既有狀態)。玩家進度存在資料庫，不在這裡。"""
    defaults = {
        "uid": None, "flash": [], "last_touch_day": None, "llm_errors": [],
        "quiz_id": None, "quiz_data": None, "quiz_answered": False, "quiz_correct": None,
        "quiz_selected": None, "quiz_reward_msg": "", "quiz_warn": False, "quiz_nonce": 0,
        "recipe_variant": 0, "recipe_out": None, "workout_variant": 0, "workout_out": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def show_flash():
    """統一顯示並清空提示訊息佇列"""
    ss = st.session_state
    items, ss.flash = list(ss.get("flash", [])), []
    for kind, msg in items:
        st.toast(msg, icon={"success": "✅", "warning": "⚠️", "error": "❌"}.get(kind, "ℹ️"))


def reset_quiz_state():
    """
    【Callback】重置測驗狀態鎖。綁定在上傳圖片 / 相機 / 輸入方式 / 語言程度的 on_change。
    Callback 會在「下一次 rerun 開始之前」執行，所以主程式讀到的一定是已重置的乾淨狀態，
    不會出現「新圖 + 舊的已作答鎖」。注意：EXP 等進度存在資料庫，這裡不會也不能被重置。
    """
    ss = st.session_state
    ss.quiz_id = None
    ss.quiz_data = None
    ss.quiz_answered = False
    ss.quiz_correct = None
    ss.quiz_selected = None
    ss.quiz_reward_msg = ""
    ss.quiz_warn = False
    ss.quiz_nonce += 1          # radio 的 key 含版本號 -> 全新元件，舊選取自動消失


def on_level_change():
    """【Callback】語言程度改變：存進資料庫 (跨裝置同步) 並重置測驗，下一輪依新程度出題"""
    ss = st.session_state
    if ss.get("uid") and ss.get("lang_level") in LANG_LEVELS:
        db_exec(update(users).where(users.c.id == ss.uid).values(lang_level=ss.lang_level))
    reset_quiz_state()


def ensure_quiz(row: pd.Series, food_df: pd.DataFrame, level: str):
    """
    確保 session_state 裡的題目與「目前食物 + 目前語言程度」一致。
    quiz_id = 食物:程度；兩者任一改變就自動重置狀態鎖並重新出題 (雙重保險，不只靠 on_change)。
    """
    ss = st.session_state
    quiz_key = f"{row['AI_Label']}:{level}"
    if ss.quiz_id != quiz_key:
        reset_quiz_state()
        ss.quiz_id = quiz_key
        with st.spinner(f"AI 出題老師正在準備【{level}】程度的題目..."):
            ss.quiz_data = generate_quiz(row, food_df, level, seed=quiz_key)


def submit_answer():
    """
    【Callback】送出答案。所有「上鎖、加分」都集中在這裡。
    防刷分三道關卡：① 已作答直接 return ② 先上鎖再判斷 ③ 資料庫複合主鍵 + 每日上限 (見 claim_quiz_reward)
    """
    ss = st.session_state
    if ss.quiz_answered or ss.quiz_data is None or not ss.uid:      # 關卡 ①
        return
    choice = ss.get(f"quiz_choice_{ss.quiz_nonce}")
    if choice is None:
        ss.quiz_warn = True                                         # 還沒選 -> 不上鎖，只提示
        return
    ss.quiz_warn, ss.quiz_answered, ss.quiz_selected = False, True, choice   # 關卡 ②：先上鎖
    ss.quiz_correct = (choice == ss.quiz_data["answer"])
    ss.quiz_reward_msg = ""
    if ss.quiz_correct:
        status = claim_quiz_reward(ss.uid, f"{today_str()}:{ss.quiz_id}")      # 關卡 ③
        if status == "ok":
            ss.quiz_reward_msg = f"🎉 獲得 +{EXP_PER_QUIZ} EXP！"
            complete_mission(ss.uid, "quiz")
            check_badges(ss.uid)
        elif status == "dup":
            ss.quiz_reward_msg = "這一題今天已經領過獎勵囉，明天或換個食物再來挑戰！"
        else:
            ss.quiz_reward_msg = f"今天的測驗獎勵已達上限（{QUIZ_DAILY_CAP} 次），明天再來吧！"


def save_meal_cb(p: dict):
    """【Callback】記錄一餐。放在 callback 而不是主程式：避免 rerun 時重複寫入"""
    if add_meal(p):
        push_flash("success", f"已記錄 {p['meal_type']}：{p['name']}（{p['kcal'] * 1:.0f} kcal）")
        complete_mission(p["uid"], "log_meal")
        check_badges(p["uid"])
    else:
        push_flash("warning", "這張照片的這一餐已經記錄過了（可先刪除舊紀錄再重記）")


def delete_meal_cb(uid: int, meal_id: int):
    delete_meal(uid, meal_id)
    push_flash("success", "已刪除這筆紀錄")


def claim_story_cb(uid: int):
    if not complete_mission(uid, "story"):
        push_flash("info", "今天的故事任務已經完成囉")
    check_badges(uid)


def like_cb(uid: int, post_id: int):
    toggle_like(uid, post_id)


def delete_post_cb(uid: int, post_id: int):
    delete_post(uid, post_id)
    push_flash("success", "貼文已刪除")


def share_post_cb(uid: int, kind: str, title: str, body: str):
    err = create_post(uid, kind, title, body, "public", None)
    if err:
        push_flash("warning", err)
    else:
        push_flash("success", "已分享到社群（公開）")
        check_badges(uid)


def logout_cb():
    ss = st.session_state
    for k in ("lang_level", "recipe_out", "workout_out"):
        ss.pop(k, None)
    ss.uid, ss.last_touch_day = None, None
    reset_quiz_state()


# ==========================================
# 6. 健康計算與提醒工具
# ==========================================
def compute_energy(user: dict, weight):
    """
    Mifflin-St Jeor 公式估算 BMR / TDEE / 每日目標熱量。資料不足回傳 None。
    保護機制：目標熱量不低於下限；BMI 過低時不套用減脂赤字。
    """
    if not (user.get("age") and user.get("height_cm") and weight):
        return None
    sex = user.get("sex") or "不指定"
    base = 10 * weight + 6.25 * user["height_cm"] - 5 * user["age"]
    bmr = base + 5 if sex == "男" else base - 161 if sex == "女" else base - 78
    tdee = bmr * ACTIVITY_LEVELS.get(user.get("activity"), 1.2)
    bmi = weight / ((user["height_cm"] / 100) ** 2)
    delta = GOALS.get(user.get("goal"), 0)
    note = ""
    if delta < 0 and bmi < 18.5:
        delta, note = 0, "你的 BMI 偏低，不建議減脂，已改以維持體重計算。"
    floor = KCAL_FLOOR.get(sex, 1350)
    target = tdee + delta
    if target < floor:
        target, note = floor, f"為了健康，每日目標熱量不低於 {floor} kcal。"
    return {"bmr": bmr, "tdee": tdee, "target": target, "bmi": bmi, "note": note}


def bmi_label(bmi: float) -> str:      # 依台灣國健署成人 BMI 標準
    return "過輕" if bmi < 18.5 else "正常" if bmi < 24 else "過重" if bmi < 27 else "肥胖"


def load_remind(user: dict) -> dict:
    try:
        cfg = json.loads(user.get("remind_json") or "{}")
    except Exception:
        cfg = {}
    return {**DEFAULT_REMIND, **cfg}


def build_ics(remind: dict) -> str:
    """產生每日重複的行事曆提醒檔 (.ics)，匯入手機行事曆後由系統推播提醒 (Streamlit 本身無法主動推播)"""
    now_utc = datetime.now(timezone.utc)
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Bite and Learn//TW", "CALSCALE:GREGORIAN"]
    for meal in ("早餐", "午餐", "晚餐"):
        t = remind.get(meal)
        if not t:
            continue
        hh, mm = map(int, t.split(":"))
        start = datetime.combine(_now().date(), dtime(hh, mm), tzinfo=TZ_TAIPEI).astimezone(timezone.utc)
        lines += ["BEGIN:VEVENT", f"UID:bitelearn-{meal}-{hh:02d}{mm:02d}@bitelearn",
                  f"DTSTAMP:{now_utc:%Y%m%dT%H%M%SZ}", f"DTSTART:{start:%Y%m%dT%H%M%SZ}", "DURATION:PT15M",
                  "RRULE:FREQ=DAILY", f"SUMMARY:記錄{meal} - Bite & Learn", "DESCRIPTION:打開 Bite & Learn 拍照記錄這一餐",
                  "BEGIN:VALARM", "TRIGGER:PT0M", "ACTION:DISPLAY", "DESCRIPTION:記錄餐點", "END:VALARM", "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)


def render_reminder_banner(user: dict):
    """站內提醒：已過設定時間 3 小時內、還沒記錄該餐 -> 顯示提醒"""
    cfg = load_remind(user)
    if not cfg.get("enabled"):
        return
    now = _now()
    logged = set(meals_between(user["id"], today_str(), today_str())["meal_type"])
    due = []
    for meal in ("早餐", "午餐", "晚餐"):
        try:
            hh, mm = map(int, cfg[meal].split(":"))
        except Exception:
            continue
        t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if t <= now <= t + timedelta(hours=3) and meal not in logged:
            due.append(meal)
    if due:
        st.warning(f"🔔 記錄提醒：「{'、'.join(due)}」還沒有記錄，拍張照片吧！")


# ==========================================
# 7. 側邊欄與登入畫面
# ==========================================
def render_auth_screen():
    st.markdown('<div class="main-title">🥗 Bite & Learn 跨界識食</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">拍下餐點 → AI 辨識與衛教 → 順便學中日英三語。登入後資料可在不同裝置同步。</div>',
                unsafe_allow_html=True)
    if db_is_local():
        st.info("ℹ️ 目前使用本機 SQLite 資料庫（適合測試）。要真正跨裝置同步、重啟不遺失，請在 secrets 設定 DATABASE_URL。")
    t_login, t_reg = st.tabs(["🔑 登入", "🆕 註冊"])
    with t_login:
        with st.form("login_form"):
            nick = st.text_input("暱稱")
            pw = st.text_input("密碼", type="password")
            go = st.form_submit_button("登入", type="primary")
        if go:
            uid, err = login_user(nick, pw)
            if err:
                st.error(err)
            else:
                st.session_state.uid = uid
                st.rerun()
    with t_reg:
        with st.form("register_form"):
            nick = st.text_input("暱稱（2~20 字，之後會顯示在社群）")
            pw = st.text_input("密碼（至少 6 個字元）", type="password")
            pw2 = st.text_input("再輸入一次密碼", type="password")
            go = st.form_submit_button("建立帳號", type="primary")
        if go:
            if pw != pw2:
                st.error("兩次輸入的密碼不一致")
            else:
                uid, err = register_user(nick, pw)
                if err:
                    st.error(err)
                else:
                    st.session_state.uid = uid
                    st.rerun()
    st.caption("🔒 密碼以加鹽雜湊 (PBKDF2) 儲存，系統不會保存明文；連續輸錯 5 次會暫時鎖定。")


def render_player_dashboard(user: dict):
    """玩家儀表板：連續登入、等級、EXP 進度條、每日任務、徽章 (資料全部來自資料庫)"""
    exp = user["exp"] or 0
    st.subheader("🎮 玩家儀表板")
    c1, c2 = st.columns(2)
    c1.metric("🔥 連續登入", f"{user['streak'] or 0} 天")
    c2.metric("⭐ 等級", f"Lv.{exp // EXP_PER_LEVEL + 1}")
    st.progress((exp % EXP_PER_LEVEL) / EXP_PER_LEVEL, text=f"EXP {exp % EXP_PER_LEVEL} / {EXP_PER_LEVEL}（累積 {exp}）")
    done = missions_done_today(user["id"])
    st.markdown("**📋 今日任務**")
    for key, (name, exp_gain) in MISSIONS.items():
        st.markdown(f"{'✅' if key in done else '⬜'} {name}　`+{exp_gain} EXP`")
    have = user_badges(user["id"])
    if have:
        st.markdown("**🏅 徽章**　" + "　".join(BADGES[b][0] for b in have if b in BADGES))


def render_api_diagnostics():
    with st.expander("🔧 API 診斷"):
        st.caption(f"模型：`{GEMINI_MODEL_NAME}`｜gTTS：{'已安裝' if GTTS_OK else '未安裝（使用瀏覽器語音）'}｜"
                   f"資料庫：{get_engine().dialect.name}")
        if st.button("測試 Gemini 連線", key="api_ping"):
            try:
                st.success(f"連線成功：{llm_model.generate_content('請只回覆：OK').text.strip()[:40]}")
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


def render_sidebar(user: dict):
    """回傳 (圖片 bytes, 圖片簽章, 語言程度)"""
    ss = st.session_state
    with st.sidebar:
        st.title("🥗 Bite & Learn")
        st.markdown(f"👤 **{md_escape(user['nickname'])}**")
        render_player_dashboard(user)
        st.markdown("---")

        st.subheader("🌐 語言程度")
        if ss.get("lang_level") not in LANG_LEVELS:
            ss.lang_level = user.get("lang_level") if user.get("lang_level") in LANG_LEVELS else "零基礎"
        level = st.selectbox("選擇你的外語程度", LANG_LEVELS, key="lang_level", on_change=on_level_change,
                             help="會影響外語例句、食物故事與測驗題的難度")
        st.markdown("---")

        st.subheader("📥 影像輸入")
        method = st.radio("請選擇輸入來源：", ["📁 上傳圖片", "📷 開啟相機"], key="input_method",
                          on_change=reset_quiz_state)
        data = None
        if "上傳" in method:
            f = st.file_uploader("選擇一張食物照片", type=["jpg", "jpeg", "png", "webp"],
                                 key="uploader_widget", on_change=reset_quiz_state)   # ★ 換圖 -> 重置測驗鎖
        else:
            f = st.camera_input("對準食物拍攝", key="camera_widget", on_change=reset_quiz_state)
        if f is not None:
            data = f.getvalue()
        sig = hashlib.md5(data).hexdigest()[:12] if data else None

        st.markdown("---")
        render_api_diagnostics()
        st.button("🚪 登出", on_click=logout_cb)
        st.caption("Powered by Streamlit, YOLOv8 & Google Gemini")
    return data, sig, level


# ==========================================
# 8. 各頁籤
# ==========================================
def default_meal_type() -> str:
    h = _now().hour
    return "早餐" if h < 10 else "午餐" if h < 15 else "晚餐" if h < 21 else "點心/宵夜"


def pick_food(food_df: pd.DataFrame, ctx: dict, prefix: str) -> pd.Series:
    """學習/測驗頁籤用的食物選單：預設為剛辨識到的食物，也可自己挑別的來學 (不一定要拍照)"""
    labels = list(food_df["AI_Label"])
    default = ctx["known"][0][0] if ctx["known"] else labels[0]
    names = {r.AI_Label: f"{r.ZH_Word} {r.EN_Word}" for r in food_df.itertuples()}
    chosen = st.selectbox("想學哪一種食物？", labels, index=labels.index(default),
                          format_func=lambda l: names[l], key=f"{prefix}_{ctx['sig'] or 'none'}")
    return food_row(food_df, chosen)


def render_tab_record(user: dict, food_df: pd.DataFrame, ctx: dict):
    """⚡ 快速健康紀錄：辨識 → AI 選項與說明 → 記錄這餐 → 今日與近 7 天統計"""
    uid, today, sig = user["id"], today_str(), ctx["sig"] or "none"
    MANUAL = "✏️ 其他（手動輸入）"
    left, right = st.columns(2, gap="large")

    with left:
        card_header("📷 [AI 視覺辨識]")
        if ctx["detection"]:
            top = ctx["known"][0] if ctx["known"] else None
            cap = f"標籤: {top[0]} | 置信度: {top[1]:.1%}" if top else "沒有辨識到資料庫內的食物"
            st.image(ctx["detection"]["plotted"], caption=cap, use_container_width=True)
            if ctx["unknown"]:
                st.caption("AI 還看到：" + "、".join(md_escape(x) for x in ctx["unknown"]) + "（資料庫尚未收錄，可手動輸入）")
            if not ctx["known"]:
                st.warning("⚠️ 找不到資料庫內的食物，請從右側手動選擇或輸入。")
        else:
            st.info("👈 從左側上傳照片或開啟相機；也可以直接在右側選擇食物來記錄。")

    with right:
        card_header("🍽️ [選擇你吃的食物]")
        if ctx["known"]:
            confs = dict(ctx["known"])
            opts = [l for l, _ in ctx["known"]] + [MANUAL]
            st.caption("AI 找到以下候選，請選你實際吃的那一個：")
        else:
            confs = {}
            opts = list(food_df["AI_Label"]) + [MANUAL]

        def _fmt(l):
            if l == MANUAL:
                return MANUAL
            r = food_row(food_df, l)
            c = f"｜信心度 {confs[l]:.0%}" if l in confs else ""
            return f"{traffic_emoji(r['Traffic_Light'])} {r['ZH_Word']} {r['EN_Word']}（約 {r['Kcal']:.0f} kcal/份{c}）"

        choice = st.radio("食物", opts, format_func=_fmt, key=f"cand_{sig}", label_visibility="collapsed")

        if choice == MANUAL:
            name = st.text_input("食物名稱", key=f"mname_{sig}", max_chars=40)
            base_kcal = st.number_input("每份熱量 (kcal)", 0, 3000, 250, step=10, key=f"mkcal_{sig}")
            tl = st.selectbox("你覺得它是…", ["🟢 綠燈 (自訂)", "🟡 黃燈 (自訂)", "🔴 紅燈 (自訂)"], index=1, key=f"mtl_{sig}")
            label, traffic, source = "custom", tl, "manual"
            disp_name = name.strip()
        else:
            r = food_row(food_df, choice)
            label, traffic, source = choice, str(r["Traffic_Light"]), "ai"
            base_kcal, disp_name = float(r["Kcal"]), f"{r['ZH_Word']} {r['EN_Word']}"

    # ---- 食科紅綠燈 + AI 說明 + 更健康的選擇 ----
    st.markdown("---")
    c1, c2 = st.columns(2, gap="large")
    emo = traffic_emoji(traffic)
    with c1:
        card_header("🚦 [食科紅綠燈]")
        box = {"🔴": st.error, "🟡": st.warning, "🟢": st.success}[emo]
        box(f"### {traffic}")
        st.markdown({"🔴": "⚠️ **營養評價**：高熱量、高鈉或過度加工，建議控管頻率。",
                     "🟡": "⚖️ **營養評價**：適量食用，搭配其他食物均衡攝取。",
                     "🟢": "🌿 **營養評價**：富含優質營養素或膳食纖維，推薦！"}[emo])
        if emo != "🟢" and source == "ai":
            alts = food_df[(food_df["Traffic_Light"].map(traffic_emoji) == "🟢") & (food_df["AI_Label"] != label)]
            alts = alts.sort_values("Kcal").head(3)
            if not alts.empty:
                st.markdown("💡 **更健康的選擇**：" + "、".join(
                    f"{a.ZH_Word}（{a.Kcal:.0f} kcal）" for a in alts.itertuples()))
    with c2:
        card_header("🩺 [生醫 AI 動態分析]")
        if source == "ai":
            with st.spinner("生醫系 AI 正在為您生成專屬分析..."):
                warning = generate_dynamic_warning(r["EN_Word"])
            st.info(f"**🔬 來自生醫系 AI 的專屬提醒：**\n\n{md_escape(warning)}")
            st.caption("※ 本衛教內容由 Gemini AI 生成，僅供日常健康生活管理參考。")
        else:
            st.info("手動輸入的食物沒有 AI 分析；你仍可記錄熱量。")

    # ---- 記錄這一餐 ----
    st.markdown("---")
    card_header("📝 [記錄這一餐]")
    f1, f2, f3 = st.columns(3)
    meal_type = f1.selectbox("餐別", MEAL_TYPES, index=MEAL_TYPES.index(default_meal_type()), key="meal_type_sel")
    portion = f2.slider("份量（幾份）", 0.5, 3.0, 1.0, 0.5, key="portion_sel")
    kcal = float(base_kcal) * portion
    f3.metric("估計熱量", f"{kcal:.0f} kcal")

    energy = compute_energy(user, latest_weight(uid))
    today_df = meals_between(uid, today, today)
    consumed = float(today_df["kcal"].sum()) if not today_df.empty else 0.0
    if energy:
        st.progress(min((consumed + kcal) / energy["target"], 1.0),
                    text=f"今日攝取（含這餐）{consumed + kcal:.0f} / 目標 {energy['target']:.0f} kcal")
    else:
        st.caption(f"今日已攝取 {consumed:.0f} kcal（到「💪 健康管理」填寫身體資料，就能看到每日目標）")

    can_save = bool(disp_name)
    payload = {"uid": uid, "day": today, "meal_type": meal_type, "label": label, "name": disp_name or "",
               "kcal": kcal, "traffic": traffic, "portion": portion, "source": source,
               "dedupe": f"{uid}:{today}:{meal_type}:{sig}:{label}" if (ctx["sig"] and source == "ai") else None}
    st.button("✅ 記錄這餐", type="primary", on_click=save_meal_cb, args=(payload,), disabled=not can_save)
    if not can_save:
        st.caption("請先輸入食物名稱")

    # ---- 今日紀錄 ----
    st.markdown("---")
    card_header("📒 [今日紀錄]")
    today_df = meals_between(uid, today, today)
    if today_df.empty:
        st.caption("今天還沒有紀錄。")
    for r2 in today_df.itertuples():
        a, b = st.columns([6, 1])
        a.markdown(f"{traffic_emoji(r2.traffic)} **{r2.meal_type}**｜{md_escape(r2.name)} × {r2.portion:g}　`{r2.kcal:.0f} kcal`")
        b.button("🗑️", key=f"del_{r2.id}", on_click=delete_meal_cb, args=(uid, int(r2.id)), help="刪除這筆")

    # ---- 近 7 天趨勢 ----
    week_df = meals_between(uid, (_now() - timedelta(days=6)).strftime("%Y-%m-%d"), today)
    if not week_df.empty:
        st.markdown("---")
        card_header("📈 [近 7 天趨勢]")
        daily = week_df.groupby("day")["kcal"].sum()
        daily = daily.reindex([(_now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)], fill_value=0)
        st.bar_chart(daily.rename("每日熱量 (kcal)"))
        counts = week_df["traffic"].map(traffic_emoji).value_counts()
        m1, m2, m3 = st.columns(3)
        m1.metric("🟢 綠燈餐", int(counts.get("🟢", 0)))
        m2.metric("🟡 黃燈餐", int(counts.get("🟡", 0)))
        m3.metric("🔴 紅燈餐", int(counts.get("🔴", 0)))


def render_tab_language(user: dict, food_df: pd.DataFrame, ctx: dict, level: str):
    """🌍 外語文化探索：中日英三語名稱 + 發音 + 分級例句 (滑鼠懸停翻譯) + 食物故事"""
    row = pick_food(food_df, ctx, "lang_food")
    zh, en, jp = row["ZH_Word"], row["EN_Word"], row["JP_Word"]
    card_header("🎌 [中・日・英 三語對照]")
    c1, c2, c3 = st.columns(3)
    c1.metric("中文", str(zh))
    c2.metric("英語 English", str(en))
    c3.metric("日語 日本語", str(jp))
    p1, p2, p3 = st.columns(3)
    with p1:
        render_speech_panel([(str(zh), str(zh))], "zh-TW", 50)
    with p2:
        render_speech_panel([(str(en), str(en))], "en-US", 50)
    with p3:
        render_speech_panel([(clean_speech_text(jp), clean_speech_text(jp))], "ja-JP", 50)

    st.markdown("---")
    st.markdown(f"##### 📖 {level}程度例句　<small>（把滑鼠移到<u>虛線單字</u>上看翻譯）</small>", unsafe_allow_html=True)
    with st.spinner("AI 語言老師正在撰寫例句..."):
        culture, fb = generate_culture_text(str(en), str(jp), level)
    if fb:
        st.caption("⚠️ AI 例句暫時無法生成，以下為內建基礎例句。")
    lc1, lc2 = st.columns(2, gap="large")
    for col, key, title, tts, gtts_lang in ((lc1, "en", "🇬🇧 English", "en-US", "en"), (lc2, "jp", "🇯🇵 日本語", "ja-JP", "ja")):
        block = culture[key]
        with col:
            st.markdown(f"**{title}**")
            st.markdown(f'<div class="abbr-sentence">{build_abbr_html(block["sentence"], block["vocab"])}</div>',
                        unsafe_allow_html=True)         # 內容已在 build_abbr_html 內 escape
            st.caption("中文：" + md_escape(block.get("translation_zh", "")))
            items = [("整句", block["sentence"])] + [(str(v["word"]), str(v["word"])) for v in block["vocab"]
                                                    if isinstance(v, dict) and v.get("word")]
            render_speech_panel(items, tts, 90)
            render_tts_audio(block["sentence"], gtts_lang)
    st.caption("💡 手機等觸控裝置無法「懸停」，請看下方單字表。")
    with st.expander("📚 本句單字表"):
        for key, title in (("en", "English"), ("jp", "日本語")):
            vdf = pd.DataFrame([v for v in culture[key]["vocab"] if isinstance(v, dict)])
            if not vdf.empty and {"word", "meaning"} <= set(vdf.columns):
                st.markdown(f"**{title}**")
                st.dataframe(vdf[["word", "meaning"]].rename(columns={"word": "單字", "meaning": "意思"}),
                             hide_index=True, use_container_width=True)

    # ---- 食物故事 (按需載入，節省 API 額度) ----
    st.markdown("---")
    card_header("📖 [食物背後的故事與文化]")
    if st.toggle("顯示這個食物的故事（中／日／英）", key=f"story_toggle_{ctx['sig'] or 'none'}"):
        with st.spinner("AI 作家正在寫故事..."):
            story, sfb = generate_story(row, level)
        if sfb:
            st.caption("⚠️ AI 故事暫時無法生成，以下為內建小知識。")
        st.markdown(f"#### {md_escape(story['zh'].get('title', ''))}")
        st.markdown(md_escape(story["zh"]["text"]))
        s1, s2 = st.columns(2, gap="large")
        for col, key, title, tts in ((s1, "en", "🇬🇧 English", "en-US"), (s2, "ja", "🇯🇵 日本語", "ja-JP")):
            with col:
                st.markdown(f"**{title}**")
                st.markdown(f'<div class="abbr-sentence" style="font-size:1.1rem;line-height:1.9rem">'
                            f'{build_abbr_html(story[key]["text"], story[key]["vocab"])}</div>', unsafe_allow_html=True)
                render_speech_panel([("朗讀全文", story[key]["text"])], tts, 50)
        if story.get("fun_fact"):
            st.info("💡 冷知識：" + md_escape(story["fun_fact"]))
        done = "story" in missions_done_today(user["id"])
        st.button("✅ 我讀完了（領取任務獎勵）" if not done else "✅ 今日故事任務已完成", disabled=done,
                  on_click=claim_story_cb, args=(user["id"],), key=f"story_done_{ctx['sig'] or 'none'}")

    with st.expander("💡 文化小知識：點我展開隨堂提問"):
        st.markdown(f"**題目：{md_escape(row['Quiz_Question'])}**")
        if st.checkbox("🙋 查看解答", key="reveal_quiz_answer"):
            st.markdown(f"🎉 **正解：** {md_escape(row['Quiz_Ans'])}")


def render_tab_quiz(user: dict, food_df: pd.DataFrame, ctx: dict, level: str):
    """🎯 每日測驗任務：依食物與語言程度動態出題，含狀態鎖"""
    ss = st.session_state
    row = pick_food(food_df, ctx, "quiz_food")
    ensure_quiz(row, food_df, level)
    quiz = ss.quiz_data
    card_header("🎯 [每日測驗任務]")
    st.caption(f"目前程度：**{level}**（可在左側切換，題目會跟著更換）｜答對 +{EXP_PER_QUIZ} EXP｜"
               f"同一題每天只能領一次，每天最多 {QUIZ_DAILY_CAP} 次獎勵")
    if quiz.get("source") == "fallback":
        st.caption("⚠️ AI 出題暫時失敗，這是內建題庫的備援題（詳見側邊欄「API 診斷」）。")
    st.markdown(f"#### ❓ {md_escape(quiz['question'])}")
    st.radio("請選擇答案：", quiz["options"], index=None, key=f"quiz_choice_{ss.quiz_nonce}",
             disabled=ss.quiz_answered)                      # 【狀態鎖】已作答 -> 選項鎖定
    st.button("✅ 送出答案", on_click=submit_answer, disabled=ss.quiz_answered, type="primary")   # 【狀態鎖】
    if ss.quiz_warn and not ss.quiz_answered:
        st.warning("請先選擇一個答案再送出喔！")
    if ss.quiz_answered:
        if ss.quiz_correct:
            st.success(f"✅ 答對了！ {ss.quiz_reward_msg}")
        else:
            st.error(f"❌ 答錯了～正解是：**{md_escape(quiz['answer'])}**")
        if quiz.get("explanation"):
            st.info(f"📝 解說：{md_escape(quiz['explanation'])}")
        st.caption("想繼續挑戰？換一種食物或調整語言程度就會有新題目！")


TIMER_HTML = """
<div style="font-family:sans-serif;text-align:center;padding:6px">
  <div id="t" style="font-size:44px;font-weight:700;color:#0F172A">00:00</div>
  <div id="msg" style="color:#4F46E5;min-height:24px;margin:4px 0"></div>
  <button onclick="start()">▶ 開始</button> <button onclick="pauseT()">⏸ 暫停</button> <button onclick="resetT()">↺ 重設</button>
</div>
<style>button{padding:6px 14px;border:1px solid #CBD5E1;border-radius:9999px;background:#EEF2FF;color:#4F46E5;font-size:15px;cursor:pointer;margin:2px}</style>
<script>
  const total = __MIN__ * 60; let left = total, h = null;
  const el = document.getElementById('t'), msg = document.getElementById('msg');
  function fmt(s){ const m = Math.floor(s/60), r = s%60; return String(m).padStart(2,'0')+':'+String(r).padStart(2,'0'); }
  function say(t){ msg.textContent = t;
    if ('speechSynthesis' in window){ speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(t); u.lang='zh-TW'; speechSynthesis.speak(u);} }
  function render(){ el.textContent = fmt(left); }
  function start(){ if (h) return; if (left === total) say('開始囉！不用追求完美，動起來就是勝利！');
    h = setInterval(function(){ left--; render();
      if (left === Math.floor(total/2)) say('已經過一半了，你超棒的！');
      if (left === 60) say('最後一分鐘，撐住！');
      if (left <= 0){ clearInterval(h); h = null; say('完成！太厲害了，記得回來記錄這次運動喔！'); } }, 1000); }
  function pauseT(){ if (h){ clearInterval(h); h = null; say('休息一下沒關係，準備好再繼續。'); } }
  function resetT(){ clearInterval(h); h = null; left = total; render(); msg.textContent=''; }
  render();
</script>
"""


def render_tab_health(user: dict, food_df: pd.DataFrame):
    """💪 健康管理：代謝計算 / 體重追蹤 / 食譜推薦 / 運動菜單 / 運動陪伴"""
    ss, uid = st.session_state, user["id"]
    weight = latest_weight(uid)
    energy = compute_energy(user, weight)
    st.caption("⚕️ 本頁數值為一般估算，不能取代醫師或營養師的建議；若有進食困擾或身體不適，請尋求專業協助。")
    t_meta, t_w, t_rec, t_ex, t_buddy = st.tabs(["🔥 代謝與目標", "⚖️ 體重追蹤", "🍳 食譜推薦", "🏃 運動菜單", "🤝 運動陪伴"])

    with t_meta:
        with st.form("profile_form"):
            a, b, c = st.columns(3)
            sexes = ["男", "女", "不指定"]
            sex = a.selectbox("生理性別（影響代謝公式）", sexes, index=sexes.index(user["sex"]) if user["sex"] in sexes else 2)
            age = b.number_input("年齡", 10, 100, int(user["age"] or 20))
            height = c.number_input("身高 (cm)", 120.0, 220.0, float(user["height_cm"] or 165.0), step=0.5)
            d, e, f = st.columns(3)
            wt = d.number_input("目前體重 (kg)", 30.0, 200.0, float(weight or 60.0), step=0.1)
            acts, goals = list(ACTIVITY_LEVELS), list(GOALS)
            act = e.selectbox("活動量", acts, index=acts.index(user["activity"]) if user["activity"] in acts else 0)
            goal = f.selectbox("目標", goals, index=goals.index(user["goal"]) if user["goal"] in goals else 0)
            saved = st.form_submit_button("💾 儲存並計算", type="primary")
        if saved:
            db_exec(update(users).where(users.c.id == uid).values(sex=sex, age=int(age), height_cm=float(height),
                                                                  activity=act, goal=goal))
            save_weight(uid, wt)
            push_flash("success", "已儲存身體資料")
            st.rerun()
        if energy:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("基礎代謝 BMR", f"{energy['bmr']:.0f} kcal")
            m2.metric("每日總消耗 TDEE", f"{energy['tdee']:.0f} kcal")
            m3.metric("建議每日攝取", f"{energy['target']:.0f} kcal")
            m4.metric("BMI", f"{energy['bmi']:.1f}", bmi_label(energy["bmi"]), delta_color="off")
            if energy["note"]:
                st.warning(energy["note"])
            consumed = float(meals_between(uid, today_str(), today_str())["kcal"].sum())
            st.progress(min(consumed / energy["target"], 1.0), text=f"今日已攝取 {consumed:.0f} / {energy['target']:.0f} kcal")
            st.caption("公式：Mifflin-St Jeor；BMI 分級依台灣國健署成人標準。")
        else:
            st.info("填寫並儲存身體資料後，這裡會顯示你的代謝與每日目標。")

    with t_w:
        a, b = st.columns([2, 1])
        new_w = a.number_input("今天的體重 (kg)", 30.0, 200.0, float(weight or 60.0), step=0.1, key="weight_input")
        b.markdown("<br>", unsafe_allow_html=True)
        if b.button("儲存今日體重"):
            save_weight(uid, new_w)
            st.success("已儲存")
        hist = weight_history(uid)
        if hist.empty:
            st.caption("還沒有體重紀錄。")
        else:
            st.line_chart(hist.set_index("day")["kg"].rename("體重 (kg)"))
            diff = hist["kg"].iloc[-1] - hist["kg"].iloc[0]
            st.metric("目前體重", f"{hist['kg'].iloc[-1]:.1f} kg", f"{diff:+.1f} kg（相較第一筆）", delta_color="off")

    with t_rec:
        r1, r2, r3, r4 = st.columns(4)
        meal = r1.selectbox("餐別", MEAL_TYPES[:3], key="rec_meal")
        restr = r2.text_input("飲食限制（如：素食、不吃牛）", key="rec_restr", max_chars=40)
        mins = r3.slider("烹調時間（分鐘）", 5, 60, 15, key="rec_min")
        equip = r4.selectbox("設備", ["宿舍（電鍋／微波爐）", "一般廚房", "只有超商"], key="rec_eq")
        per_meal = energy["target"] / 3 if energy else None
        if st.button("🍳 推薦食譜", type="primary"):
            ss.recipe_variant += 1
            with st.spinner("營養師 AI 構思中..."):
                text, err = generate_recipes(per_meal, user["goal"] or "維持體重", meal, restr, mins, equip, ss.recipe_variant)
            ss.recipe_out = (text, err, meal)
        if ss.recipe_out:
            text, err, meal_used = ss.recipe_out
            if err:
                st.error(err)
            else:
                st.markdown(text)
                st.button("📤 分享到社群（公開）", key="share_recipe", on_click=share_post_cb,
                          args=(uid, "🍳 食譜", f"AI 推薦的{meal_used}食譜", text[:1900]))

    with t_ex:
        x1, x2, x3 = st.columns(3)
        mins = x1.slider("運動時間（分鐘）", 10, 90, 30, 5, key="ex_min")
        equip = x2.selectbox("器材", ["無器材（徒手）", "啞鈴／彈力帶", "健身房", "戶外跑步／走路"], key="ex_eq")
        inten = x3.selectbox("強度", ["輕鬆", "中等", "有挑戰"], index=1, key="ex_int")
        if st.button("🏃 產生運動菜單", type="primary"):
            ss.workout_variant += 1
            with st.spinner("教練 AI 設計中..."):
                text, err = generate_workout(user["goal"] or "維持體重", mins, equip, inten, ss.workout_variant)
            ss.workout_out = (text, err)
        if ss.workout_out:
            text, err = ss.workout_out
            if err:
                st.error(err)
            else:
                st.markdown(text)
                st.button("📤 分享到社群（公開）", key="share_workout", on_click=share_post_cb,
                          args=(uid, "🏃 運動", "AI 運動菜單", text[:1900]))

    with t_buddy:
        card_header("🤝 [運動陪伴]")
        tmin = st.slider("這次想運動幾分鐘？", 5, 90, 20, 5, key="buddy_min")
        components.html(TIMER_HTML.replace("__MIN__", str(int(tmin))), height=170)
        st.caption("計時器會在開始、過半、最後一分鐘與完成時為你加油（需開啟瀏覽器聲音）。完成後請在下方記錄。")
        with st.form("workout_form", clear_on_submit=True):
            k1, k2 = st.columns(2)
            kind = k1.selectbox("運動種類", list(EXERCISES_MET))
            minutes = k2.number_input("實際運動分鐘", 1, 300, int(tmin))
            ok = st.form_submit_button("💾 記錄這次運動", type="primary")
        if ok:
            kcal = EXERCISES_MET[kind] * float(weight or 60.0) * minutes / 60
            add_workout(uid, kind, minutes, kcal)
            push_flash("success", f"已記錄 {kind} {minutes} 分鐘，約消耗 {kcal:.0f} kcal")
            complete_mission(uid, "move")
            check_badges(uid)
            st.rerun()
        wdf = workouts_since(uid, (_now() - timedelta(days=6)).strftime("%Y-%m-%d"))
        if not wdf.empty:
            m1, m2 = st.columns(2)
            m1.metric("近 7 天運動", f"{int(wdf['minutes'].sum())} 分鐘")
            m2.metric("估計消耗", f"{wdf['kcal'].sum():.0f} kcal")
            days = [(_now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
            st.bar_chart(wdf.groupby("day")["minutes"].sum().reindex(days, fill_value=0).rename("運動分鐘"))


def render_tab_community(user: dict):
    """👥 社群：食譜 / 運動 / 成果分享 (公開或群組)"""
    uid = user["id"]
    t_feed, t_new, t_group = st.tabs(["📰 動態", "✍️ 發文", "👥 我的群組"])
    groups_mine = my_groups(uid)

    with t_new:
        with st.form("post_form", clear_on_submit=True):
            kind = st.selectbox("類型", POST_KINDS)
            title = st.text_input("標題", max_chars=60)
            body = st.text_area("內容", max_chars=2000, height=160)
            attach = st.checkbox("附上我今天的熱量摘要")
            vis = st.radio("誰可以看？", ["公開", "群組"], horizontal=True)
            gsel = st.selectbox("群組（選「群組」時使用）", [g["id"] for g in groups_mine] or [None],
                                format_func=lambda gid: next((g["name"] for g in groups_mine if g["id"] == gid), "（尚未加入群組）"))
            go = st.form_submit_button("發佈", type="primary")
        if go:
            text = body
            if attach:
                tot = float(meals_between(uid, today_str(), today_str())["kcal"].sum())
                text += f"\n\n（今日攝取約 {tot:.0f} kcal）"
            err = create_post(uid, kind, title, text, "public" if vis == "公開" else "group", gsel)
            if err:
                st.error(err)
            else:
                push_flash("success", "發佈成功！")
                check_badges(uid)
                st.rerun()
        st.caption("🛡️ 請勿張貼個資或不當內容；內容以純文字顯示。")

    def _render_feed(scope: str):
        items = fetch_posts(scope, uid)
        if not items:
            st.caption("目前沒有貼文。")
        for p in items:
            with st.container(border=True):
                st.markdown(f"**{p['kind']}　{md_escape(p['title'])}**")
                where = "公開" if p["visibility"] == "public" else f"群組：{md_escape(p['group_name'] or '')}"
                st.caption(f"{md_escape(p['nickname'])}｜{p['ts'][:16]}｜{where}")
                st.markdown(md_escape(p["body"]))
                c1, c2 = st.columns([1, 5])
                c1.button(f"{'❤️' if p['liked'] else '🤍'} {p['likes']}", key=f"like_{scope}_{p['id']}",
                          on_click=like_cb, args=(uid, p["id"]))
                if p["user_id"] == uid:
                    c2.button("🗑️ 刪除", key=f"delp_{scope}_{p['id']}", on_click=delete_post_cb, args=(uid, p["id"]))

    with t_feed:
        scope_label = st.radio("看哪裡？", ["公開", "我的群組", "我的貼文"], horizontal=True, key="feed_scope")
        _render_feed({"公開": "public", "我的群組": "groups", "我的貼文": "mine"}[scope_label])

    with t_group:
        g1, g2 = st.columns(2, gap="large")
        with g1:
            with st.form("create_group_form", clear_on_submit=True):
                gname = st.text_input("建立新群組", max_chars=30, placeholder="例如：減脂打卡小隊")
                if st.form_submit_button("建立"):
                    code, err = create_group(uid, gname)
                    push_flash("error" if err else "success", err or f"群組已建立，邀請碼：{code}")
                    st.rerun()
        with g2:
            with st.form("join_group_form", clear_on_submit=True):
                code_in = st.text_input("用邀請碼加入", max_chars=12)
                if st.form_submit_button("加入"):
                    err = join_group(uid, code_in)
                    push_flash("error" if err else "success", err or "已加入群組")
                    st.rerun()
        for g in groups_mine:
            st.markdown(f"**{md_escape(g['name'])}**　邀請碼 `{g['code']}`　成員 {g['members']} 人"
                        + ("　👑 你是群主" if g["owner_id"] == uid else ""))
        if not groups_mine:
            st.caption("你還沒有加入任何群組。")


def render_tab_settings(user: dict):
    """⚙️ 帳號與設定：記錄提醒 / 資料匯出 / 密碼 / 刪除帳號"""
    uid = user["id"]
    cfg = load_remind(user)
    card_header("🔔 [記錄提醒]")
    with st.form("remind_form"):
        enabled = st.toggle("開啟站內提醒（在設定時間後尚未記錄該餐，就會在頁面上方提示）", value=bool(cfg["enabled"]))
        cols = st.columns(3)
        times = {}
        for col, meal in zip(cols, ("早餐", "午餐", "晚餐")):
            hh, mm = map(int, cfg[meal].split(":"))
            times[meal] = col.time_input(f"{meal}提醒", dtime(hh, mm), key=f"rt_{meal}")
        if st.form_submit_button("儲存提醒設定", type="primary"):
            new_cfg = {"enabled": enabled, **{m: t.strftime("%H:%M") for m, t in times.items()}}
            db_exec(update(users).where(users.c.id == uid).values(remind_json=json.dumps(new_cfg)))
            push_flash("success", "提醒設定已儲存")
            st.rerun()
    st.download_button("📅 下載手機行事曆提醒 (.ics)", build_ics(cfg).encode("utf-8"),
                       file_name="bite_learn_reminders.ics", mime="text/calendar")
    st.caption("網頁無法主動推播通知；匯入 .ics 到 Google／Apple 行事曆後，手機會在每天固定時間提醒你記錄。")

    st.markdown("---")
    card_header("🔄 [跨裝置同步與資料]")
    if db_is_local():
        st.warning("目前使用本機 SQLite：換裝置無法同步，且 Streamlit Cloud 重啟後資料可能消失。"
                   "請在 secrets 設定 DATABASE_URL（Supabase / Neon 的 Postgres 皆可）。")
    else:
        st.success("已連接雲端資料庫：在任何裝置登入同一個帳號，資料都會同步。")
    csv = meals_between(uid, "2000-01-01", "2999-12-31").drop(columns=["user_id", "dedupe"]).to_csv(index=False)
    st.download_button("⬇️ 匯出我的飲食紀錄 (CSV)", csv.encode("utf-8-sig"), file_name="my_meals.csv", mime="text/csv")

    st.markdown("---")
    card_header("🔐 [帳號安全]")
    with st.form("pw_form", clear_on_submit=True):
        old, new = st.text_input("目前密碼", type="password"), st.text_input("新密碼（至少 6 個字元）", type="password")
        if st.form_submit_button("變更密碼"):
            err = change_password(uid, old, new)
            st.error(err) if err else st.success("密碼已更新")
    with st.expander("⚠️ 刪除帳號（無法復原）"):
        pw = st.text_input("輸入密碼確認", type="password", key="del_pw")
        sure = st.checkbox("我了解這會永久刪除我所有的紀錄、貼文與我建立的群組", key="del_sure")
        if st.button("永久刪除我的帳號", disabled=not sure):
            u = get_user(uid)
            if u and hmac.compare_digest(_hash_pw(pw or "", u["salt"]), u["pw_hash"]):
                delete_account(uid)
                logout_cb()
                st.rerun()
            else:
                st.error("密碼不正確")


# ==========================================
# 9. 主程序
# ==========================================
def main():
    init_session_state()
    ss = st.session_state
    get_engine()                                     # 確保資料表已建立

    if not ss.uid:
        render_auth_screen()
        return
    user = get_user(ss.uid)
    if user is None:                                 # 帳號已不存在 (例如在別的裝置刪除)
        logout_cb()
        st.rerun()

    # 連續登入 (Streak)：每個 session 每天只更新一次
    if ss.last_touch_day != today_str():
        touch_login(ss.uid)
        check_badges(ss.uid)
        ss.last_touch_day = today_str()
        user = get_user(ss.uid)

    food_df = load_food_database("food_database.csv")
    img_bytes, sig, level = render_sidebar(user)
    user = get_user(ss.uid)                          # sidebar 內的 callback 可能已改變 EXP，重新讀取

    st.markdown('<div class="main-title">🥗 Bite & Learn 跨界識食</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">從餐盤看見全世界：AI 辨識 ✕ 飲食紀錄 ✕ 健康管理 ✕ 中日英三語學習</div>',
                unsafe_allow_html=True)
    show_flash()
    render_reminder_banner(user)

    detection, known, unknown = None, [], []
    if img_bytes:
        with st.spinner("AI 正在仔細辨識您的餐點..."):
            detection = run_detection(img_bytes)
        known, unknown = build_candidates(detection["dets"], food_df)
        if known:
            st.success(f"🎯 AI 找到 {len(known)} 種可能的餐點：" + "、".join(
                f"**{food_row(food_df, l)['EN_Word']}**（{c:.0%}）" for l, c in known))
    ctx = {"sig": sig, "detection": detection, "known": known, "unknown": unknown}

    tabs = st.tabs(["⚡ 快速健康紀錄", "🌍 外語文化探索", "🎯 每日測驗任務", "💪 健康管理", "👥 社群", "⚙️ 帳號與設定"])
    with tabs[0]:
        render_tab_record(user, food_df, ctx)
    with tabs[1]:
        render_tab_language(user, food_df, ctx, level)
    with tabs[2]:
        render_tab_quiz(user, food_df, ctx, level)
    with tabs[3]:
        render_tab_health(user, food_df)
    with tabs[4]:
        render_tab_community(user)
    with tabs[5]:
        render_tab_settings(user)
    show_flash()


if __name__ == "__main__":
    main()
