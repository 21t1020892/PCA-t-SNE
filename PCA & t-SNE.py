import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import mlflow
import os
import time
from datetime import datetime
from mlflow.tracking import MlflowClient

# Hàm kết nối MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/21t1020892/PCA-t-SNE.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "21t1020892"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "xN8@Q7V@Pbr6CYZ"
    mlflow.set_experiment("PCA & t-SNE")
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI

# Hàm tải dữ liệu MNIST từ OpenML
@st.cache_data
def load_mnist_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32) / 255.0
    return X, y

# Tab hiển thị dữ liệu
def data():
    st.header("📘 Dữ Liệu MNIST từ OpenML")
    
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.X = None
        st.session_state.y = None

    if st.button("⬇️ Tải dữ liệu từ OpenML"):
        with st.spinner("⏳ Đang tải dữ liệu MNIST từ OpenML..."):
            X, y = load_mnist_data()
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.data_loaded = True
            st.success("✅ Dữ liệu đã được tải thành công!")

    if st.session_state.data_loaded:
        X, y = st.session_state.X, st.session_state.y
        st.write("""
            **Thông tin tập dữ liệu MNIST:**
            - Tổng số mẫu: {}
            - Kích thước mỗi ảnh: 28 × 28 pixels (784 đặc trưng)
            - Số lớp: 10 (chữ số từ 0-9)
        """.format(X.shape[0]))

        st.subheader("Một số hình ảnh mẫu")
        n_samples = 10
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        for i, idx in enumerate(indices):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(X[idx].reshape(28, 28), cmap='gray')
            axes[row, col].set_title(f"Label: {y[idx]}")
            axes[row, col].axis("off")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("ℹ️ Nhấn nút 'Tải dữ liệu từ OpenML' để tải và hiển thị dữ liệu.")

# Hàm giải thích PCA
def explain_pca():
    st.markdown("## 🧠 PCA - Phân tích Thành phần Chính")

    st.markdown("""
    **PCA (Principal Component Analysis)** là một kỹ thuật giảm chiều tuyến tính giúp chuyển dữ liệu từ không gian nhiều chiều sang không gian ít chiều hơn, đồng thời giữ lại phần lớn thông tin quan trọng (phương sai).  
    - **Mục tiêu**: Tìm các hướng chính (principal components) mà dữ liệu biến thiên nhiều nhất, sau đó chiếu dữ liệu lên các hướng này.
    - **Ứng dụng**: Trực quan hóa dữ liệu, giảm kích thước dữ liệu để tăng tốc các thuật toán học máy.
    """)

    st.markdown("### 🔹 **PCA hoạt động như thế nào?**")
    st.markdown("""
    Hãy tưởng tượng bạn có dữ liệu 2D với các điểm nằm rải rác nhưng chủ yếu phân bố theo một hướng chéo. PCA sẽ tìm hướng chính mà dữ liệu biến thiên mạnh nhất và biến đổi dữ liệu sang hệ tọa độ mới dựa trên hướng đó.
    """)

    st.markdown("## 🔹 **Các bước thực hiện PCA**")

    with st.expander("1️⃣ Tính vector kỳ vọng của toàn bộ dữ liệu"):
        st.latex(r"\bar{x} = \frac{1}{N} \sum_{n=1}^{N} x_n")
        st.write("Vector kỳ vọng giúp xác định trung tâm của dữ liệu.")

    with st.expander("2️⃣ Chuẩn hóa dữ liệu bằng cách trừ kỳ vọng"):
        st.latex(r"\hat{x}_n = x_n - \bar{x}")
        st.write("Dịch chuyển dữ liệu về gốc tọa độ giúp PCA hoạt động chính xác hơn.")

    with st.expander("3️⃣ Tính ma trận hiệp phương sai"):
        st.latex(r"S = \frac{1}{N} \hat{X} \hat{X}^T")
        st.write("Ma trận này mô tả mối quan hệ giữa các chiều dữ liệu.")

    with st.expander("4️⃣ Tính trị riêng và vector riêng"):
        st.write("Giải phương trình trị riêng:")
        st.latex(r"S v = \lambda v")
        st.write("- **Trị riêng** (\\( \lambda \\)): Mức độ biến thiên theo hướng của vector riêng.")
        st.write("- **Vector riêng** (\\( v \\)): Hướng quan trọng trong không gian dữ liệu.")

    with st.expander("5️⃣ Chọn thành phần chính và tạo không gian con"):
        st.write("Sắp xếp trị riêng theo thứ tự giảm dần và chọn \\( K \\) vector lớn nhất:")
        st.latex(r"U_K = [v_1, v_2, ..., v_K]")
        st.write("Các vector này tạo thành hệ trực giao để giảm chiều dữ liệu.")

    with st.expander("6️⃣ Chiếu dữ liệu lên không gian mới"):
        st.latex(r"Z = U_K^T \hat{X}")
        st.write("Dữ liệu mới chính là tọa độ trong không gian mới.")

    with st.expander("7️⃣ Xấp xỉ lại dữ liệu ban đầu (tùy chọn)"):
        st.latex(r"x \approx U_K Z + \bar{x}")
        st.write("Có thể tái tạo dữ liệu ban đầu gần đúng từ không gian giảm chiều.")

    st.markdown("""
    **Hình trên**: Các mũi tên đỏ là các trục chính mà PCA tìm ra. Trục dài hơn (Trục 1) là hướng có phương sai lớn nhất.
    """)

    st.markdown("### ✅ **Ưu điểm của PCA**")
    st.markdown("""
    - Giảm chiều hiệu quả, giữ được thông tin chính (phương sai lớn).
    - Tăng tốc xử lý cho các mô hình học máy.
    - Loại bỏ nhiễu bằng cách bỏ qua các chiều có phương sai nhỏ.
    """)

    st.markdown("### ❌ **Nhược điểm của PCA**")
    st.markdown("""
    - Chỉ hiệu quả với dữ liệu có cấu trúc tuyến tính.
    - Các thành phần chính không còn ý nghĩa trực quan như đặc trưng gốc.
    - Nhạy cảm với dữ liệu chưa chuẩn hóa (cần scale trước nếu các chiều có đơn vị khác nhau).
    """)
    
def explain_tsne():
    st.markdown("## 🌌 t-SNE - Giảm chiều Phi tuyến")

    st.markdown("""
    **t-SNE (t-Distributed Stochastic Neighbor Embedding)** là một kỹ thuật giảm chiều phi tuyến, tập trung vào việc bảo toàn cấu trúc cục bộ của dữ liệu (khoảng cách giữa các điểm gần nhau).  
    - **Mục tiêu**: Chuyển dữ liệu từ không gian cao chiều (ví dụ: 784 chiều của MNIST) xuống 2D hoặc 3D để trực quan hóa.
    - **Ứng dụng**: Chủ yếu dùng để khám phá và hiển thị dữ liệu phức tạp.
    """)

    st.markdown("### 🔹 **Tham số quan trọng trong t-SNE**")
    st.markdown("""
    - **`n_components`**:  
      - **Ý nghĩa**: Số chiều mà dữ liệu sẽ được giảm xuống (thường là 2 hoặc 3 để trực quan hóa).  
      - **Giá trị**: Một số nguyên dương (ví dụ: 2 cho 2D, 3 cho 3D).  
      - **Tác động**:  
        - 2 hoặc 3: Phù hợp để vẽ biểu đồ trực quan.  
        - Không hỗ trợ giá trị lớn hơn vì t-SNE chủ yếu dùng cho trực quan hóa.  
      - **Trong code này**: Bạn chọn từ 1 đến 3 để hiển thị dữ liệu dưới dạng 1D, 2D, hoặc 3D.
    """)

    st.markdown("## 🔹 **Các bước hoạt động của t-SNE**")

    with st.expander("1️⃣ Tính phân phối xác suất trong không gian cao chiều"):
        st.write("Xác suất tương đồng giữa hai điểm dữ liệu \\( x_i \\) và \\( x_j \\) được tính bằng:")
        st.latex(r"P_{j|i} = \frac{\exp(- \| x_i - x_j \|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(- \| x_i - x_k \|^2 / 2\sigma_i^2)}")
        st.write("- **\\( \sigma_i \\)**: Độ rộng của phân phối Gaussian tại điểm \\( x_i \\).")
        st.write("- **Mục tiêu**: Định nghĩa xác suất gần gũi giữa các điểm.")

    with st.expander("2️⃣ Xây dựng phân phối đối xứng"):
        st.latex(r"P_{ij} = \frac{P_{j|i} + P_{i|j}}{2N}")
        st.write("Phân phối \\( P_{ij} \\) mô tả mối quan hệ tương đồng giữa các điểm dữ liệu trong không gian gốc.")

    with st.expander("3️⃣ Xây dựng phân phối trong không gian giảm chiều"):
        st.write("Dùng phân phối **t-Student** với một bậc tự do để tính xác suất \\( Q_{ij} \\) trong không gian mới:")
        st.latex(r"Q_{ij} = \frac{(1 + \| y_i - y_j \|^2)^{-1}}{\sum_{k \neq l} (1 + \| y_k - y_l \|^2)^{-1}}")
        st.write("- **Lý do chọn t-Student**: Phân phối có đuôi dài, giúp giữ lại cấu trúc cục bộ.")

    with st.expander("4️⃣ Tính hàm mất mát Kullback-Leibler (KL) Divergence"):
        st.write("Hàm mất mát đo sự khác biệt giữa hai phân phối:")
        st.latex(r"C = \sum_{i} \sum_{j} P_{ij} \log \frac{P_{ij}}{Q_{ij}}")
        st.write("**Mục tiêu**: Làm cho \\( Q_{ij} \\) gần với \\( P_{ij} \\) nhất có thể.")

    with st.expander("5️⃣ Tối ưu hóa bằng Gradient Descent"):
        st.write("Cập nhật tọa độ \\( y_i \\) để giảm hàm mất mát:")
        st.latex(r"y_i^{(t+1)} = y_i^{(t)} + \eta \frac{\partial C}{\partial y_i}")
        st.write("- **\\( \eta \\)**: Tốc độ học (learning rate).")
        st.write("- **Sử dụng kỹ thuật Momentum để tăng tốc hội tụ.**")

    with st.expander("6️⃣ Hoàn thành và trực quan hóa"):
        st.write("- Sau một số vòng lặp, dữ liệu sẽ được ánh xạ sang không gian 2D hoặc 3D.")
        st.write("- Các điểm dữ liệu gần nhau trong không gian cao chiều sẽ vẫn gần nhau sau khi giảm chiều.")

        st.markdown("### ✅ **Ưu điểm của t-SNE**")
        st.markdown("""
        - Tạo các cụm dữ liệu rõ ràng, dễ nhìn trong không gian 2D/3D.
        - Phù hợp với dữ liệu phi tuyến tính (PCA không làm được).
        - Rất tốt để trực quan hóa dữ liệu phức tạp như MNIST.
        """)

        st.markdown("### ❌ **Nhược điểm của t-SNE**")
        st.markdown("""
        - Tốn nhiều thời gian tính toán, đặc biệt với dữ liệu lớn.
        - Nhạy cảm với cách thiết lập ban đầu (cần chọn cẩn thận).
        - Không bảo toàn cấu trúc toàn cục, chỉ tập trung vào cục bộ.
        - Không phù hợp để giảm chiều cho học máy (chỉ dùng để trực quan hóa).
        """)

# Hàm thực hiện giảm chiều và trực quan hóa
def dimensionality_reduction():
    st.title("📉 Giảm Chiều Dữ liệu MNIST")

    if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
        st.warning("⚠ Vui lòng tải dữ liệu từ tab 'Dữ Liệu' trước khi tiếp tục!")
        return

    X, y = st.session_state["X"], st.session_state["y"]
    st.write(f"Tổng số mẫu: {X.shape[0]}, Số chiều ban đầu: {X.shape[1]}")

    num_samples = st.slider("Chọn số lượng mẫu:", 1000, X.shape[0], 5000, step=1000)
    X_subset, y_subset = X[:num_samples], y[:num_samples]

    method = st.radio("Chọn phương pháp giảm chiều:", ["PCA", "t-SNE"])
    n_components = st.slider("Số chiều giảm xuống:", 1, 3, 2)

    run_name = st.text_input("🔹 Nhập tên Run:", "")  # Để trống để người dùng tự nhập

    if st.button("🚀 Chạy Giảm Chiều"):
        if not run_name:
            st.error("⚠ Vui lòng nhập tên Run trước khi tiếp tục!")
            return

        with st.spinner(f"Đang thực hiện {method}..."):
            # Khởi tạo thanh trạng thái
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Cập nhật trạng thái: Bắt đầu
            status_text.text("Bắt đầu quá trình giảm chiều...")
            progress_bar.progress(0.1)

            mlflow_input()
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("method", method)
                mlflow.log_param("n_components", n_components)
                mlflow.log_param("num_samples", num_samples)
                mlflow.log_param("original_dim", X.shape[1])

                start_time = time.time()

                if method == "PCA":
                    # Giai đoạn 1: Khởi tạo PCA
                    status_text.text("Khởi tạo PCA...")
                    reducer = PCA(n_components=n_components)
                    progress_bar.progress(0.3)

                    # Giai đoạn 2: Fit và transform dữ liệu
                    status_text.text("Đang giảm chiều dữ liệu với PCA...")
                    X_reduced = reducer.fit_transform(X_subset)
                    progress_bar.progress(0.7)
                    #explained_variance_ratio là Phương sai 
                    if n_components > 1:
                        explained_variance = np.sum(reducer.explained_variance_ratio_)
                        mlflow.log_metric("explained_variance_ratio", explained_variance)
                else:
                    # Giai đoạn 1: Khởi tạo t-SNE
                    status_text.text("Khởi tạo t-SNE...")
                    perplexity = min(30, num_samples - 1)
                    mlflow.log_param("perplexity", perplexity)
                    reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
                    progress_bar.progress(0.3)

                    # Giai đoạn 2: Fit và transform dữ liệu
                    status_text.text("Đang giảm chiều dữ liệu với t-SNE...")
                    X_reduced = reducer.fit_transform(X_subset)
                    progress_bar.progress(0.7)
                    #kl_divergence là một thước đo sự khác biệt giữa hai phân phối xác suất, được sử dụng trong t-SNE để tối ưu hóa việc giảm chiều
                    #là một thước đo sự khác biệt giữa hai phân phối xác suất, được sử dụng trong t-SNE để tối ưu hóa việc giảm chiều
                    if hasattr(reducer, "kl_divergence_"):
                        mlflow.log_metric("KL_divergence", reducer.kl_divergence_)

                # Giai đoạn 3: Trực quan hóa
                status_text.text("Đang tạo biểu đồ trực quan...")
                elapsed_time = time.time() - start_time
                mlflow.log_metric("elapsed_time", elapsed_time)

                if n_components == 1:
                    fig = px.line(x=range(len(X_reduced)), y=X_reduced.flatten(), color=y_subset,
                                  title=f"{method} giảm chiều xuống 1D",
                                  labels={'x': "Mẫu", 'y': "Giá trị thành phần"})
                elif n_components == 2:
                    fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=y_subset,
                                     title=f"{method} giảm chiều xuống 2D",
                                     labels={'x': "Thành phần 1", 'y': "Thành phần 2"})
                else:
                    fig = px.scatter_3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                                        color=y_subset,
                                        title=f"{method} giảm chiều xuống 3D",
                                        labels={'x': "Thành phần 1", 'y': "Thành phần 2", 'z': "Thành phần 3"})
                st.plotly_chart(fig)
                progress_bar.progress(0.9)

                # Giai đoạn 4: Lưu dữ liệu và hoàn tất
                status_text.text("Đang lưu dữ liệu và hoàn tất...")
                os.makedirs("logs", exist_ok=True)
                reduced_data_path = f"logs/{method}_{n_components}D_X_reduced.npy"
                np.save(reduced_data_path, X_reduced)
                mlflow.log_artifact(reduced_data_path)

                # Lưu run_name vào session_state để dùng trong tab MLflow
                st.session_state["last_run_name"] = run_name

                progress_bar.progress(1.0)
                status_text.text("Hoàn thành!")

                st.success(f"✅ Hoàn thành {method} trong {elapsed_time:.2f} giây!")
                st.markdown(f"🔗 [Xem kết quả trên MLflow]({st.session_state['mlflow_url']})")

# Hàm hiển thị thông tin MLflow Experiments (Hiển thị tất cả run_name và chi tiết run vừa chạy)
def show_experiment_selector():
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'> MLflow Experiments </h1>", unsafe_allow_html=True)
    if 'mlflow_url' in st.session_state:
        st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
    else:
        st.warning("⚠️ URL MLflow chưa được khởi tạo!")

    mlflow_input()
    experiment_name = "PCA-t-SNE"
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'🟢 Active' if selected_experiment.lifecycle_stage == 'active' else '🔴 Deleted'}")
    st.write(f"**Artifact Location:** `{selected_experiment.artifact_location}`")

    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này!", icon="🚨")
        return

    # Hiển thị tất cả run_name dưới dạng danh sách chọn
    st.subheader("🏃‍♂️ Tất cả Runs trong Experiment")
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_data = mlflow.get_run(run_id)
        run_name = run_data.info.run_name if run_data.info.run_name else f"Run_{run_id[:8]}"
        run_info.append((run_name, run_id))

    # Loại bỏ trùng lặp trong danh sách run_name để hiển thị trong selectbox
    run_name_to_id = dict(run_info)  # Từ điển ánh xạ run_name -> run_id (giữ run_id cuối cùng nếu trùng)
    run_names = list(run_name_to_id.keys())  # Danh sách run_name không trùng lặp

    st.write("**Chọn Run để xem chi tiết:**")
    selected_run_name = st.selectbox("Danh sách Run Names:", run_names, key="run_selector")

    # Tìm run tương ứng với run_name được chọn
    selected_run_id = run_name_to_id[selected_run_name]
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.markdown(f"<h3 style='color: #28B463;'>📌 Chi tiết Run Được Chọn: {selected_run_name}</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("#### ℹ️ Thông tin cơ bản")
            st.info(f"**Run Name:** {selected_run_name}")
            st.info(f"**Run ID:** `{selected_run_id}`")
            st.info(f"**Trạng thái:** {selected_run.info.status}")
            start_time_ms = selected_run.info.start_time
            if start_time_ms:
                start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
            else:
                start_time = "Không có thông tin"
            st.info(f"**Thời gian chạy:** {start_time}")

        with col2:
            params = selected_run.data.params
            if params:
                st.write("#### ⚙️ Parameters")
                with st.container(height=200):
                    st.json(params)

            metrics = selected_run.data.metrics
            if metrics:
                st.write("#### 📊 Metrics")
                with st.container(height=200):
                    st.json(metrics)

    # Hiển thị chi tiết run vừa chạy từ tab Giảm Chiều (nếu có)
    st.markdown("---")
    if "last_run_name" not in st.session_state or not st.session_state["last_run_name"]:
        st.warning("⚠ Chưa có run nào được thực hiện gần đây. Vui lòng chạy giảm chiều trong tab 'Giảm Chiều' để xem chi tiết!")
    else:
        last_run_name = st.session_state["last_run_name"]
        st.subheader(f"📌 Chi tiết Run Gần Đây: {last_run_name}")

        # Tìm run với run_name vừa chạy
        selected_last_run = None
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            run_data = mlflow.get_run(run_id)
            if run_data.info.run_name == last_run_name:
                selected_last_run = run_data
                selected_last_run_id = run_id
                break

        if selected_last_run:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("#### ℹ️ Thông tin cơ bản")
                st.info(f"**Run Name:** {last_run_name}")
                st.info(f"**Run ID:** `{selected_last_run_id}`")
                st.info(f"**Trạng thái:** {selected_last_run.info.status}")
                start_time_ms = selected_last_run.info.start_time
                if start_time_ms:
                    start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    start_time = "Không có thông tin"
                st.info(f"**Thời gian chạy:** {start_time}")

            with col2:
                params = selected_last_run.data.params
                if params:
                    st.write("#### ⚙️ Parameters")
                    with st.container(height=200):
                        st.json(params)

                metrics = selected_last_run.data.metrics
                if metrics:
                    st.write("#### 📊 Metrics")
                    with st.container(height=200):
                        st.json(metrics)
        else:
            st.warning(f"⚠ Không tìm thấy run với tên '{last_run_name}'. Vui lòng kiểm tra lại hoặc chạy lại trong tab 'Giảm Chiều'!")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Powered by Streamlit & MLflow</p>", unsafe_allow_html=True)

# Giao diện chính
def main():
    st.title("🚀 MNIST Dimensionality Reduction with PCA & t-SNE")
    tabs = st.tabs(["📘 Dữ Liệu", "📘 PCA", "📘 t-SNE", "📉 Giảm Chiều", "📊 MLflow"])

    with tabs[0]:
        data()
    with tabs[1]:
        explain_pca()
    with tabs[2]:
        explain_tsne()
    with tabs[3]:
        dimensionality_reduction()
    with tabs[4]:
        show_experiment_selector()

if __name__ == "__main__":
    main()