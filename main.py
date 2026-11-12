import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DEFAULT_CSV = "ecommerce_orders_clean.csv"

st.set_page_config(page_title="주문 금액 예측기", page_icon="🛒", layout="centered")
st.title("🛒 주문 금액 예측기")
st.caption("가격(price)과 수량(quantity)으로 주문 금액을 예측하는 선형 회귀 모델")


@st.cache_data
def load_data(source):
    df = pd.read_csv(source)
    df = df.dropna(subset=["price", "quantity"]).copy()
    # 주문금액 = 가격 x 수량
    df["order_amount"] = df["price"] * df["quantity"]

    # 이상치 제거 (IQR 방법): 주문금액이 Q3 + 1.5*IQR 보다 크거나 Q1 - 1.5*IQR 보다 작은 행 삭제
    n_before = len(df)
    df_raw = df.copy()  # 이상치 제거 전 전체 데이터 (분석 그래프용)
    q1 = df["order_amount"].quantile(0.25)
    q3 = df["order_amount"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df[(df["order_amount"] >= lower) & (df["order_amount"] <= upper)].copy()
    return df, n_before, lower, upper, df_raw


# ---------- 데이터 불러오기 ----------
source = None
if os.path.exists(DEFAULT_CSV):
    source = DEFAULT_CSV
else:
    uploaded = st.file_uploader(
        f"'{DEFAULT_CSV}' 파일을 찾을 수 없습니다. CSV 파일을 업로드해 주세요.", type="csv"
    )
    if uploaded is not None:
        source = uploaded

if source is None:
    st.info("데이터 파일을 기다리는 중입니다.")
    st.stop()

df, n_before, out_lower, out_upper, df_raw = load_data(source)

missing = {"price", "quantity"} - set(df.columns)
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}")
    st.stop()


# ---------- 모델 학습 ----------
@st.cache_resource
def train_model(data: pd.DataFrame):
    X = data[["price", "quantity"]]
    y = data["order_amount"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return (
        model,
        r2_score(y_test, pred),
        mean_absolute_error(y_test, pred),
        len(X_train),
        len(X_test),
        y_test.to_numpy(),  # 테스트 데이터의 실제 주문금액 (그래프용)
        pred,  # 테스트 데이터의 예측값 (그래프용)
    )


model, r2, mae, n_train, n_test, y_test_actual, y_test_pred = train_model(df)

# ---------- 슬라이더 범위: 데이터의 최솟값 ~ 최댓값 ----------
p_min, p_max = float(df["price"].min()), float(df["price"].max())
q_min, q_max = int(df["quantity"].min()), int(df["quantity"].max())
p_default = float(np.clip(df["price"].median(), p_min, p_max))
q_default = int(np.clip(df["quantity"].median(), q_min, q_max))
p_step = max(round((p_max - p_min) / 1000, 2), 0.01)

# ---------- 품목 추가/삭제 ----------
if "n_items" not in st.session_state:
    st.session_state.n_items = 1


def add_item():
    st.session_state.n_items += 1


def remove_item():
    if st.session_state.n_items > 1:
        i = st.session_state.n_items - 1
        # 삭제되는 품목의 슬라이더 상태 정리
        st.session_state.pop(f"price_{i}", None)
        st.session_state.pop(f"qty_{i}", None)
        st.session_state.n_items -= 1


st.subheader("품목 입력")
b1, b2, b3 = st.columns([1, 1, 4])
b1.button("➕", on_click=add_item, help="품목 추가", use_container_width=True)
b2.button(
    "➖",
    on_click=remove_item,
    help="마지막 품목 삭제",
    disabled=st.session_state.n_items <= 1,
    use_container_width=True,
)
b3.markdown(f"**현재 품목 수: {st.session_state.n_items}개**")

rows = []
for i in range(st.session_state.n_items):
    with st.container(border=True):
        st.markdown(f"**품목 {i + 1}**")
        price = st.slider(
            "가격",
            min_value=p_min,
            max_value=p_max,
            value=p_default,
            step=p_step,
            key=f"price_{i}",
        )
        quantity = st.slider(
            "수량",
            min_value=q_min,
            max_value=q_max,
            value=q_default,
            step=1,
            key=f"qty_{i}",
        )
        rows.append({"품목": f"품목 {i + 1}", "price": price, "quantity": quantity})

# ---------- 예측 ----------
items = pd.DataFrame(rows)
items["예측 주문금액"] = model.predict(items[["price", "quantity"]])
items["계산값(가격×수량)"] = items["price"] * items["quantity"]

st.subheader("예측 결과")
total_pred = items["예측 주문금액"].sum()
total_calc = items["계산값(가격×수량)"].sum()

c1, c2 = st.columns(2)
c1.metric("총 예측 주문 금액", f"{total_pred:,.2f}")
c2.metric(
    "총 계산값 (가격×수량)",
    f"{total_calc:,.2f}",
    delta=f"{total_pred - total_calc:,.2f} (오차)",
)

st.dataframe(
    items.rename(columns={"price": "가격", "quantity": "수량"}).set_index("품목"),
    use_container_width=True,
)

# ---------- 그래프 ----------
st.subheader("📊 그래프")

# 그래프 1: 카테고리별 총 주문금액 (막대그래프)
st.markdown("**① 카테고리별 총 주문금액**")
if "category" in df.columns:
    cat_total = (
        df.groupby("category")["order_amount"]
        .sum()
        .sort_values(ascending=False)
        .rename("총 주문금액")
    )
    st.bar_chart(cat_total)
    st.dataframe(cat_total.round(0).astype(int).to_frame(), use_container_width=True)
else:
    st.info("category 열이 없어서 이 그래프를 그릴 수 없어요.")

# 그래프 2: 테스트 데이터의 실제값 vs 예측값 (산점도)
st.markdown("**② 실제 주문금액 vs 모델 예측값 (테스트 데이터)**")
compare = pd.DataFrame(
    {"실제 주문금액": y_test_actual, "예측 주문금액": y_test_pred}
)
st.scatter_chart(compare, x="실제 주문금액", y="예측 주문금액")
st.caption("점이 왼쪽 아래에서 오른쪽 위로 한 줄에 가깝게 모일수록 예측이 정확해요.")

# ---------- 분석 그래프 ----------
st.subheader("🔍 분석 그래프")

# 분석 그래프 1: 가격 구간별 평균 수량
st.markdown("**③ 가격 구간별 평균 수량**")
st.caption("이상치를 제거하기 전 전체 데이터로 계산했어요. (큰 금액 주문이 빠지면 가격과 수량 관계가 왜곡돼요.)")
edges = np.linspace(df_raw["price"].min(), df_raw["price"].max(), 6)  # 5개 구간
bin_labels = [
    f"{i + 1}구간 ({edges[i]:.0f}~{edges[i + 1]:.0f})" for i in range(5)
]
price_bin = pd.cut(df_raw["price"], bins=edges, labels=bin_labels, include_lowest=True)
qty_by_price = df_raw.groupby(price_bin, observed=True)["quantity"].mean().rename("평균 수량")
st.bar_chart(qty_by_price)
q_high, q_low = qty_by_price.idxmax(), qty_by_price.idxmin()
q_gap = qty_by_price.max() - qty_by_price.min()
st.caption(
    f"해석: 평균 수량이 가장 많은 구간은 {q_high}({qty_by_price.max():.2f}개), "
    f"가장 적은 구간은 {q_low}({qty_by_price.min():.2f}개)로 차이는 {q_gap:.2f}개예요. "
    "가격이 올라도 수량이 크게 줄지 않아요."
    if q_gap < 0.3
    else f"해석: 평균 수량이 가장 많은 구간은 {q_high}, 가장 적은 구간은 {q_low}로 "
    f"차이가 {q_gap:.2f}개예요. 가격에 따라 수량이 달라지는 경향이 있어요."
)

# 분석 그래프 2: 고객 유형별 평균 주문금액
st.markdown("**④ 고객 유형별 평균 주문금액**")
if "customer_segment" in df.columns:
    seg_avg = (
        df.groupby("customer_segment")["order_amount"]
        .mean()
        .sort_values(ascending=False)
        .rename("평균 주문금액")
    )
    st.bar_chart(seg_avg)
    s_high, s_low = seg_avg.idxmax(), seg_avg.idxmin()
    s_gap_pct = (seg_avg.max() - seg_avg.min()) / seg_avg.min() * 100
    st.caption(
        f"해석: 평균 주문금액이 가장 높은 고객 유형은 {s_high}({seg_avg.max():,.1f}), "
        f"가장 낮은 유형은 {s_low}({seg_avg.min():,.1f})이고, 차이는 약 {s_gap_pct:.1f}%예요."
        + (" 유형별 차이가 크지 않아요." if s_gap_pct < 10 else " 유형별로 차이가 있어요.")
    )
else:
    st.info("customer_segment 열이 없어서 이 그래프를 그릴 수 없어요.")

# ---------- 모델 성능 평가 (테스트 데이터에서 무작위 100개) ----------
st.subheader("🧪 모델 성능 평가")
st.caption("테스트 데이터에서 무작위로 100개를 뽑아, 모델의 예측이 얼마나 맞는지 계산해요.")

if "eval_seed" not in st.session_state:
    st.session_state.eval_seed = 0
if "eval_show_all" not in st.session_state:
    st.session_state.eval_show_all = False


def resample():
    st.session_state.eval_seed += 1


def toggle_table():
    st.session_state.eval_show_all = not st.session_state.eval_show_all


e1, e2 = st.columns(2)
e1.button("🔄 다시 뽑기", on_click=resample, use_container_width=True)
e2.button(
    "📂 100개 전체 보기" if not st.session_state.eval_show_all else "📁 표 닫기(앞 10행만)",
    on_click=toggle_table,
    use_container_width=True,
)

rng = np.random.default_rng(st.session_state.eval_seed)
pick = rng.choice(len(y_test_actual), size=min(100, len(y_test_actual)), replace=False)
ev_actual = y_test_actual[pick]
ev_pred = y_test_pred[pick]

ev_table = pd.DataFrame(
    {
        "실제값": ev_actual,
        "예측값": ev_pred,
        "오차(실제-예측)": ev_actual - ev_pred,
        "절댓값": np.abs(ev_actual - ev_pred),
        "제곱": (ev_actual - ev_pred) ** 2,
    }
).round(2)
ev_table.index = range(1, len(ev_table) + 1)

st.dataframe(
    ev_table if st.session_state.eval_show_all else ev_table.head(10),
    use_container_width=True,
)
if not st.session_state.eval_show_all:
    st.caption("표는 앞 10행만 보여 주고 있어요. 계산은 100개 전체로 했어요.")

ev_mae = float(np.abs(ev_actual - ev_pred).mean())
ev_mse = float(((ev_actual - ev_pred) ** 2).mean())
ev_ss_model = float(((ev_actual - ev_pred) ** 2).sum())
ev_ss_mean = float(((ev_actual - ev_actual.mean()) ** 2).sum())
ev_r2 = 1 - ev_ss_model / ev_ss_mean

k1, k2, k3 = st.columns(3)
k1.metric("MAE (평균 절대 오차)", f"{ev_mae:,.2f}")
k2.metric("MSE (평균 제곱 오차)", f"{ev_mse:,.2f}")
k3.metric("R²", f"{ev_r2:.4f}")
st.caption(
    f"R² = 1 - (모델의 제곱오차 합 {ev_ss_model:,.0f} ÷ 평균값 기준의 제곱오차 합 {ev_ss_mean:,.0f}) = {ev_r2:.4f}"
)

# ---------- 모델 정보 ----------
with st.expander("모델 성능 및 계수"):
    st.write(
        f"데이터 분할: 학습용 {n_train:,}행(50%) / 테스트용 {n_test:,}행(50%)"
    )
    m1, m2 = st.columns(2)
    m1.metric("R² (테스트)", f"{r2:.4f}")
    m2.metric("MAE (테스트)", f"{mae:,.2f}")
    st.write(
        f"절편: {model.intercept_:,.4f} / 가격 계수: {model.coef_[0]:,.4f} / 수량 계수: {model.coef_[1]:,.4f}"
    )
    st.caption(
        "주문 금액은 가격×수량(곱셈)이라 순수 선형 모델로는 오차가 생길 수 있습니다."
    )

with st.expander("데이터 미리보기"):
    st.write(f"행 수: {len(df):,}")
    st.dataframe(df[["price", "quantity", "order_amount"]].head(50))

with st.expander("데이터 전처리 (이상치 제거)"):
    st.write(
        f"IQR 방법으로 주문금액이 {out_upper:,.2f}보다 큰 주문을 이상치로 보고 삭제했어요."
    )
    n_removed = n_before - len(df)
    p1, p2, p3 = st.columns(3)
    p1.metric("전처리 전", f"{n_before:,}행")
    p2.metric("삭제된 이상치", f"{n_removed:,}행")
    p3.metric("전처리 후", f"{len(df):,}행")
