import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st

st.title("📦 배달 위치 군집화 (K-Means)")

# 데이터 로드
try:
    df = pd.read_csv("https://raw.githubusercontent.com/jdh123/delivery-clustering/main/Delivery.csv")
except Exception as e:
    st.error(f"❌ 데이터 파일을 불러올 수 없습니다: {e}")
    st.stop()

X = df[['Latitude', 'Longitude']]

# Elbow Method를 통한 최적의 K 찾기
inertias = []
K = range(1, 11)
for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertias.append(model.inertia_)

# 최적의 K 값 계산
optimal_k = 3  # 기본값 설정
if len(inertias) > 1:
    deltas = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    optimal_k = deltas.index(max(deltas)) + 1

st.write(f"📌 최적의 K 값은: {optimal_k} 개")

# 최종 KMeans 모델 학습 및 예측
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# 시각화
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="Longitude", y="Latitude", hue="Cluster", palette="tab10", ax=ax)
plt.title("📍 클러스터링 결과")
st.pyplot(fig)
