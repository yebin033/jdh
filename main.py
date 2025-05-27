import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st

st.title("📦 배달 위치 군집화 (K-Means)")

# Load data
df = pd.read_csv("Delivery.csv")
X = df[['Latitude', 'Longitude']]

# Elbow Method
inertias = []
K = range(1, 11)
for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertias.append(model.inertia_)

optimal_k = inertias.index(min(inertias[1:], key=lambda x: abs(x - inertias[0]) * 0.2)) + 1
st.write(f"📌 최적의 K 값은: {optimal_k} 개")

# Final clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Plotting
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="Longitude", y="Latitude", hue="Cluster", palette="tab10", ax=ax)
plt.title("📍 클러스터링 결과")
st.pyplot(fig)
df = pd.read_csv("https://raw.githubusercontent.com/사용자이름/저장소이름/브랜치이름/Delivery.csv")

