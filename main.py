from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="주문 금액 예측", page_icon="🛒")

# ---------------------------------------------------------
# 1. 데이터 읽기 (같은 폴더의 CSV 파일)
# ---------------------------------------------------------
CSV_PATH = Path(__file__).parent / "ecommerce_orders_clean.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    # 가격(price) x 양(quantity) = 주문금액 (새로 만드는 열)
    df["주문금액"] = df["price"] * df["quantity"]
    return df


# ---------------------------------------------------------
# 2. 학습용/테스트용으로 나누고 선형회귀 모델 학습
# ---------------------------------------------------------
@st.cache_resource
def train_model(df):
    X = df[["price", "quantity"]]  # 독립변수(재료)
    y = df["주문금액"]  # 종속변수(정답)

    # 학습용 80%, 테스트용 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)  # 학습용 데이터로만 공부시키기
    return model, len(X_train), len(X_test)


df = load_data()
model, n_train, n_test = train_model(df)

# ---------------------------------------------------------
# 3. 화면 만들기
# ---------------------------------------------------------
st.title("🛒 주문 금액 예측")
st.write("가격과 양을 고르면, 선형회귀 모델이 예상 주문금액을 알려줘요.")

# 슬라이더 범위는 데이터의 최솟값 ~ 최댓값
price_min = float(df["price"].min())
price_max = float(df["price"].max())
qty_min = int(df["quantity"].min())
qty_max = int(df["quantity"].max())

price = st.slider(
    "가격 (price)",
    min_value=price_min,
    max_value=price_max,
    value=float(round(df["price"].mean(), 2)),
    step=0.01,
)
quantity = st.slider(
    "양 (quantity)",
    min_value=qty_min,
    max_value=qty_max,
    value=int(df["quantity"].median()),
    step=1,
)

# 예측하기
input_df = pd.DataFrame([[price, quantity]], columns=["price", "quantity"])
predicted = float(model.predict(input_df)[0])
predicted = max(predicted, 0.0)  # 금액이 음수로 나오지 않게

st.metric("예상 주문금액", f"${predicted:,.2f}")

st.caption(
    f"전체 {len(df):,}건 중 학습용 {n_train:,}건, 테스트용 {n_test:,}건으로 나누어 작업했어요."
)
