import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

def run_kmeans_clustering(df, num_clusters, max_iter):
    # chuẩn hóa data
    scaler = StandardScaler()
    data = scaler.fit_transform(df)

    # huấn luyện k means
    kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iter, random_state=42)
    labels = kmeans.fit_predict(data)

    # lấy tâm cụm
    clus_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    return labels, clus_centers

# df = pd.read_csv("Iris.csv")
df=load_iris()['data']
labels, clus_center = run_kmeans_clustering(df, 3, 1000)

print(f"Ma trận kết quả phân cụm: \n{labels}")
print("Tọa độ các tâm cụm: ")
for i, center in enumerate(clus_center):
    print(f"Cụm {i}: {center}")



