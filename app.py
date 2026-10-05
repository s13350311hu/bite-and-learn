# -*- coding: utf-8 -*-
"""
Bite & Learn 跨界識食 - 智慧衛教 APP
======================================
技術棧架構：
  - 前端與互動：Streamlit (自訂卡片 CSS、響應式四宮格)
  - 電腦視覺：Ultralytics YOLOv8 (yolov8n.pt 預訓練模型)
  - 資料處理：Pandas, Pillow (PIL), NumPy
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import google.generativeai as genai

# 從 Streamlit Cloud 的 Secrets 安全讀取金鑰並設定 Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
# 使用 Gemini 1.5 Flash 模型，反應速度最快適合網頁互動
llm_model = genai.GenerativeModel('gemini-1.5-flash')
from ultralytics import YOLO

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
# 3. 側邊欄互動模組 (Sidebar Navigation)
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.title("🥗 Bite & Learn 跨界識食")
        st.markdown(
            """
            **跨領域智慧衛教系統**
            結合 **AI 電腦視覺**、**食品科學紅綠燈**、**生醫風險警語** 與 **外語文化微學習**。
            
            拍下你的餐點，一秒掌握健康密碼與趣味豆知識！
            ---
            """
        )

        st.subheader("📥 選擇影像輸入方式")
        input_method = st.radio(
            label="請選擇輸入來源：",
            options=["📁 上傳圖片 (File Uploader)", "📷 開啟相機 (Camera Input)"],
            index=0
        )

        input_image = None
        if "📁 上傳圖片" in input_method:
            uploaded_file = st.file_uploader(
                label="選擇一張食物照片",
                type=["jpg", "jpeg", "png", "webp"],
                help="支援常見格式如 JPG、PNG、WEBP"
            )
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file)
        else:
            camera_file = st.camera_input(
                label="對準食物拍攝照片",
                help="需允許瀏覽器存取相機進行拍攝"
            )
            if camera_file is not None:
                input_image = Image.open(camera_file)

        st.markdown("---")
        st.caption("Powered by Streamlit & Ultralytics YOLOv8")

    return input_image
@st.cache_data(show_spinner="生醫系 AI 正在為您生成專屬分析...")
def generate_dynamic_warning(food_name):
    """將辨識出的食物名稱丟給 Gemini，動態生成生醫警語"""
    prompt = f"""
    你現在是一位充滿幽默感、具備醫學與營養學知識的生醫系大學生。
    使用者剛剛用系統掃描到準備吃「{food_name}」。
    請用繁體中文，用大約 50 到 80 字的一小段話，給予健康警告或營養提示。
    語氣要生動活潑、有點像在吐槽或關心朋友，讓大學生看了會有共鳴。
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "⚠️ AI 護理師暫時去喝水了，請注意飲食均衡喔！"

# ==========================================
# 4. 主程序與四宮格卡片呈現
# ==========================================
def main():
    # 載入模型與衛教資料庫
    model = load_yolo_model("yolov8n.pt")
    food_df = load_food_database("food_database.csv")

    # 取得側邊欄影像輸入
    image = render_sidebar()

    # 主主畫面標題與導言
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
    bio_warning = row["Bio_Warning"]
    quiz_question = row["Quiz_Question"]
    quiz_ans = row["Quiz_Ans"]

    st.success(f"🎯 成功辨識餐點：**{en_word}** (`{detected_label}`)！信心度：**{best_conf:.1%}**")

    # ==========================================
    # 四宮格卡片排版 (2 x 2 佈局)
    # ==========================================
    col1, col2 = st.columns(2, gap="large")

    # 第一列左：📷 [AI 視覺辨識]
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

    # 第一列右：🚦 [食科紅綠燈]
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

    col3, col4 = st.columns(2, gap="large")

    # 第二列左：🩺 [生醫警語] (Gemini 升級版)
    with col3:
        st.markdown("""
        <div class="card-box">
            <div class="card-header">🩺 [生醫 AI 動態分析]</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 呼叫剛才寫好的函式，傳入食物名稱，讓 AI 即興發揮
        dynamic_warning = generate_dynamic_warning(en_word)
        
        st.info(f"**🔬 來自生醫系 AI 的專屬提醒：**\n\n{dynamic_warning}")
        st.caption("※ 本衛教內容由 Gemini AI 生成，僅供日常健康生活管理參考。")

    # 第二列右：🎌 [外語與文化微學習]
    with col4:
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

        st.markdown("---")
        # 下拉式互動測驗 (Expander)
        with st.expander("💡 點我展開隨堂文化測驗！"):
            st.markdown(f"**題目：{quiz_question}**")
            show_ans = st.checkbox("🙋 查看解答", key="reveal_quiz_answer")
            if show_ans:
                st.markdown(f"🎉 **正解：** :green[**{quiz_ans}**]")


# ==========================================
# 程式進入點
# ==========================================
if __name__ == "__main__":
    main()
