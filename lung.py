import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import os

# =========================
# [안전] 구글 무료 웹폰트 다운로드 및 설정
# =========================
@st.cache_data
def load_font():
    # Noto Sans KR 폰트 파일 URL
    font_url = "https://github.com"
    font_name = "NotoSansKR.ttf"
    
    # 서버 환경에 폰트가 없다면 다운로드
    if not os.path.exists(font_name):
        response = requests.get(font_url)
        with open(font_name, "wb") as f:
            f.write(response.content)
    return font_name

# Matplotlib용 폰트 설정
font_path = load_font()
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 페이지 설정 및 웹 화면 폰트 적용
# =========================
st.set_page_config(
    page_title="폐암 환자 군집 분석",
    page_icon="🫁",
    layout="centered"
)

# 스트림릿 웹 화면 전체에 구글 웹폰트 반영
st.markdown("""
    <style>
    @import url('https://googleapis.com');
    html, body, [data-testid="stSidebar"], .stButton, .stMarkdown, p, h1, h2, h3 {
        font-family: 'Noto Sans KR', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# 모델 & 스케일러 불러오기
# =========================
model = joblib.load("lung_model.pkl")
scaler = joblib.load("lung_scaler.pkl")

# =========================
# 학습 데이터 불러오기 (시각화용)
# =========================
df = pd.read_csv("lung.csv")
X = df[['나이', '흡연량', '음주량']]

# 스케일링 및 기존 데이터 군집 예측
X_scaled = scaler.transform(X)
df['cluster'] = model.predict(X_scaled)

# =========================
# 제목 및 UI 레이아웃
# =========================
st.title("🫁 폐암 환자 군집 분석 시스템")
st.markdown("""
AI가 환자의 특성을 분석하여  
어떤 군집(유형)에 속하는지 예측합니다.
""")
st.divider()

st.subheader("📋 환자 정보 입력")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("나이", min_value=0.0, max_value=120.0, value=50.0)
with col2:
    smoking = st.number_input("흡연량", min_value=0.0, value=10.0)
with col3:
    alcohol = st.number_input("음주량", min_value=0.0, value=5.0)

st.divider()

# =========================
# 예측 및 시각화 결과 출력
# =========================
if st.button("🔍 군집 분석하기", use_container_width=True):
    # 새로운 환자 데이터 예측
    new_patient = pd.DataFrame([[age, smoking, alcohol]], columns=['나이', '흡연량', '음주량'])
    new_patient_scaled = scaler.transform(new_patient)
    pred_cluster = model.predict(new_patient_scaled)
    cluster_num = pred_cluster[0]

    st.success(f"이 환자는 {cluster_num}번 군집에 속합니다.")
    st.write("0번은 매우 건강군, 1번은 위험군, 2번은 건강군입니다.")

    # 차트 시각화
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['흡연량'], df['음주량'], c=df['cluster'], cmap='viridis')
    
    # 사용자 위치 표시
    ax.scatter(smoking, alcohol, s=300, marker='*', color='red', label='해당 환자')

    # 구글 폰트를 차트 라벨에 적용
    ax.set_xlabel("흡연량", fontproperties=fontprop, fontsize=12)
    ax.set_ylabel("음주량", fontproperties=fontprop, fontsize=12)
    ax.set_title("군집 시각화", fontproperties=fontprop, fontsize=15)
    ax.legend(prop=fontprop)

    st.pyplot(fig)
