from __future__ import annotations

import os
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


st.set_page_config(
    page_title="Instacart Customer Segmentation",
    layout="wide",
)

DEFAULT_DESKTOP_DATA_DIR = Path("/Users/aarushi/Desktop/Dissertation/5663439_Copy")
DATA_DIRS = [
    Path(os.getenv("INSTACART_DATA_DIR", "data")),
    DEFAULT_DESKTOP_DATA_DIR,
]
CSV_FILES = {
    "orders": "orders.csv",
    "order_products_prior": "order_products__prior.csv",
    "order_products_train": "order_products__train.csv",
    "products": "products.csv",
    "aisles": "aisles.csv",
    "departments": "departments.csv",
}

REPRESENTATIONS = ["Original standardized", "PCA", "t-SNE", "UMAP", "Autoencoder"]
CLUSTER_METHODS = ["MiniBatch K-Means", "Fuzzy C-Means", "Gaussian Mixture", "Agglomerative"]

NUMERIC_FEATURES = [
    "total_orders",
    "average_days_between_orders",
    "reorder_rate",
    "average_order_hour",
    "average_order_dow",
    "unique_products",
    "average_cart_position",
    "average_basket_size",
]


@st.cache_data(show_spinner=False)
def make_demo_data(seed: int = 42) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_users = 450
    n_products = 90
    departments = pd.DataFrame(
        {
            "department_id": range(1, 8),
            "department": [
                "produce",
                "dairy eggs",
                "beverages",
                "snacks",
                "frozen",
                "pantry",
                "bakery",
            ],
        }
    )
    aisles = pd.DataFrame(
        {
            "aisle_id": range(1, 15),
            "aisle": [
                "fresh fruits",
                "fresh vegetables",
                "yogurt",
                "milk",
                "water seltzer",
                "chips pretzels",
                "ice cream",
                "canned goods",
                "bread",
                "breakfast",
                "packaged cheese",
                "tea",
                "cookies cakes",
                "pasta sauce",
            ],
        }
    )
    aisle_department = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
        8: 6,
        9: 7,
        10: 6,
        11: 2,
        12: 3,
        13: 4,
        14: 6,
    }
    product_names = [
        "Organic Bananas",
        "Bag of Organic Bananas",
        "Organic Strawberries",
        "Large Lemon",
        "Organic Baby Spinach",
        "Organic Hass Avocado",
        "Limes",
        "Organic Whole Milk",
        "Sparkling Water",
        "Organic Raspberries",
    ]
    products = []
    for product_id in range(1, n_products + 1):
        aisle_id = int(rng.integers(1, 15))
        base_name = product_names[(product_id - 1) % len(product_names)]
        products.append(
            {
                "product_id": product_id,
                "product_name": f"{base_name} {product_id}" if product_id > 10 else base_name,
                "aisle_id": aisle_id,
                "department_id": aisle_department[aisle_id],
            }
        )
    products = pd.DataFrame(products)

    orders_rows = []
    prior_rows = []
    train_rows = []
    order_id = 1
    for user_id in range(1, n_users + 1):
        orders_per_user = int(rng.integers(4, 22))
        loyalty = rng.beta(2.2, 2.8)
        favorite_products = rng.choice(products["product_id"], size=12, replace=False)
        for order_number in range(1, orders_per_user + 1):
            eval_set = "train" if order_number == orders_per_user else "prior"
            order_size = int(np.clip(rng.poisson(9 + 6 * loyalty), 3, 35))
            order_dow = int(rng.integers(0, 7))
            order_hour = int(np.clip(rng.normal(13, 4), 0, 23))
            days_since = np.nan if order_number == 1 else int(rng.integers(2, 31))
            orders_rows.append(
                {
                    "order_id": order_id,
                    "user_id": user_id,
                    "eval_set": eval_set,
                    "order_number": order_number,
                    "order_dow": order_dow,
                    "order_hour_of_day": order_hour,
                    "days_since_prior_order": days_since,
                }
            )
            basket = []
            for _ in range(order_size):
                if rng.random() < loyalty:
                    basket.append(int(rng.choice(favorite_products)))
                else:
                    basket.append(int(rng.choice(products["product_id"])))
            seen = set()
            target_rows = train_rows if eval_set == "train" else prior_rows
            for position, product_id in enumerate(dict.fromkeys(basket), start=1):
                reordered = int(product_id in seen or rng.random() < loyalty)
                seen.add(product_id)
                target_rows.append(
                    {
                        "order_id": order_id,
                        "product_id": product_id,
                        "add_to_cart_order": position,
                        "reordered": reordered,
                    }
                )
            order_id += 1

    return {
        "orders": pd.DataFrame(orders_rows),
        "order_products_prior": pd.DataFrame(prior_rows),
        "order_products_train": pd.DataFrame(train_rows),
        "products": products,
        "aisles": aisles,
        "departments": departments,
    }


def read_uploaded_or_local(name: str, uploaded_files: Dict[str, object]) -> Optional[pd.DataFrame]:
    filename = CSV_FILES[name]
    if filename in uploaded_files:
        return pd.read_csv(uploaded_files[filename])
    for data_dir in DATA_DIRS:
        path = data_dir / filename
        if path.exists():
            return pd.read_csv(path)
    return None


def find_backend_data_dir() -> Optional[Path]:
    for data_dir in DATA_DIRS:
        if all((data_dir / filename).exists() for filename in CSV_FILES.values()):
            return data_dir
    return None


@st.cache_data(show_spinner="Loading Instacart CSV files...")
def load_local_data() -> Dict[str, Optional[pd.DataFrame]]:
    return {name: read_uploaded_or_local(name, {}) for name in CSV_FILES}


def load_uploaded_data(files: Iterable[object]) -> Dict[str, Optional[pd.DataFrame]]:
    uploaded_files = {file.name: file for file in files}
    return {name: read_uploaded_or_local(name, uploaded_files) for name in CSV_FILES}


def clean_frames(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    cleaned = {}
    for name, frame in frames.items():
        df = frame.copy()
        df.columns = df.columns.str.strip()
        cleaned[name] = df

    orders = cleaned["orders"]
    orders["days_since_prior_order"] = orders["days_since_prior_order"].fillna(0)

    int_columns = {
        "orders": ["order_id", "user_id", "order_number", "order_dow", "order_hour_of_day"],
        "products": ["product_id", "aisle_id", "department_id"],
        "aisles": ["aisle_id"],
        "departments": ["department_id"],
        "order_products_prior": ["order_id", "product_id", "add_to_cart_order", "reordered"],
        "order_products_train": ["order_id", "product_id", "add_to_cart_order", "reordered"],
    }
    for frame_name, columns in int_columns.items():
        for column in columns:
            if column in cleaned[frame_name].columns:
                cleaned[frame_name][column] = cleaned[frame_name][column].astype(int)
    orders["days_since_prior_order"] = orders["days_since_prior_order"].astype(float)
    return cleaned


@st.cache_data(show_spinner="Preparing product and customer features...")
def prepare_analysis(frames: Dict[str, pd.DataFrame], order_source: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = clean_frames(frames)
    orders = frames["orders"]
    products = frames["products"]
    aisles = frames["aisles"]
    departments = frames["departments"]

    if order_source == "Train orders only":
        order_products = frames["order_products_train"].copy()
    elif order_source == "Prior orders only":
        order_products = frames["order_products_prior"].copy()
    else:
        order_products = pd.concat(
            [frames["order_products_prior"], frames["order_products_train"]],
            ignore_index=True,
        )

    product_catalog = products.merge(aisles, on="aisle_id", how="left").merge(
        departments, on="department_id", how="left"
    )
    merged = (
        orders.merge(order_products, on="order_id", how="inner")
        .merge(product_catalog, on="product_id", how="left")
    )

    basket_sizes = order_products.groupby("order_id").size().rename("basket_size").reset_index()
    order_level = orders.merge(basket_sizes, on="order_id", how="left")
    order_level["basket_size"] = order_level["basket_size"].fillna(0)

    customer_features = merged.groupby("user_id").agg(
        total_orders=("order_number", "max"),
        average_days_between_orders=("days_since_prior_order", "mean"),
        reorder_rate=("reordered", "mean"),
        average_order_hour=("order_hour_of_day", "mean"),
        average_order_dow=("order_dow", "mean"),
        unique_products=("product_id", "nunique"),
        average_cart_position=("add_to_cart_order", "mean"),
    )
    average_basket_size = order_level.groupby("user_id")["basket_size"].mean().rename("average_basket_size")
    customer_features = customer_features.join(average_basket_size, how="left").fillna(0)
    customer_features = customer_features.reset_index()

    favorite_departments = (
        merged.groupby(["user_id", "department"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["user_id", "count"], ascending=[True, False])
        .drop_duplicates("user_id")
        .rename(columns={"department": "top_department"})
    )
    favorite_aisles = (
        merged.groupby(["user_id", "aisle"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["user_id", "count"], ascending=[True, False])
        .drop_duplicates("user_id")
        .rename(columns={"aisle": "top_aisle"})
    )
    customer_features = customer_features.merge(
        favorite_departments[["user_id", "top_department"]], on="user_id", how="left"
    ).merge(favorite_aisles[["user_id", "top_aisle"]], on="user_id", how="left")

    return orders, merged, customer_features


def missing_data_message(missing: Iterable[str]) -> None:
    st.warning(
        "Missing CSVs: "
        + ", ".join(CSV_FILES[name] for name in missing)
        + ". Upload them in the sidebar or add them to the data folder."
    )


def metric_card(label: str, value: str, help_text: Optional[str] = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def format_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def render_overview(orders: pd.DataFrame, merged: pd.DataFrame, customer_features: pd.DataFrame) -> None:
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        metric_card("Orders", format_number(orders["order_id"].nunique()))
    with kpi2:
        metric_card("Customers", format_number(orders["user_id"].nunique()))
    with kpi3:
        metric_card("Products Bought", format_number(merged["product_id"].nunique()))
    with kpi4:
        metric_card("Reorder Rate", f"{merged['reordered'].mean():.1%}")
    with kpi5:
        metric_card("Avg Basket Size", f"{merged.groupby('order_id').size().mean():.1f}")

    left, right = st.columns((1.2, 1))
    with left:
        max_orders = orders.groupby("user_id")["order_number"].max().reset_index()
        fig = px.histogram(
            max_orders,
            x="order_number",
            nbins=30,
            labels={"order_number": "Total orders per customer"},
            title="Customer Order Frequency",
            color_discrete_sequence=["#277da1"],
        )
        fig.update_layout(yaxis_title="Customers", bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        hourly = orders["order_hour_of_day"].value_counts().sort_index().reset_index()
        hourly.columns = ["hour", "orders"]
        fig = px.bar(
            hourly,
            x="hour",
            y="orders",
            title="Orders by Hour",
            labels={"hour": "Hour of day", "orders": "Orders"},
            color="orders",
            color_continuous_scale="Teal",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        dow = orders["order_dow"].value_counts().sort_index().reset_index()
        dow.columns = ["day", "orders"]
        day_names = {
            0: "Sun",
            1: "Mon",
            2: "Tue",
            3: "Wed",
            4: "Thu",
            5: "Fri",
            6: "Sat",
        }
        dow["day_name"] = dow["day"].map(day_names)
        fig = px.bar(
            dow,
            x="day_name",
            y="orders",
            title="Orders by Day of Week",
            labels={"day_name": "Day", "orders": "Orders"},
            color_discrete_sequence=["#43aa8b"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(
            orders,
            x="days_since_prior_order",
            nbins=31,
            title="Days Since Prior Order",
            labels={"days_since_prior_order": "Days since prior order"},
            color_discrete_sequence=["#f3722c"],
        )
        fig.update_layout(yaxis_title="Orders", bargap=0.04)
        st.plotly_chart(fig, use_container_width=True)


def render_product_views(merged: pd.DataFrame) -> None:
    left, right = st.columns((1, 1))
    with left:
        top_n = st.slider("Top product count", 5, 30, 10, key="top_products")
    with right:
        organic_case_sensitive = st.toggle("Case-sensitive organic match", value=False)

    top_products = merged["product_name"].value_counts().head(top_n).reset_index()
    top_products.columns = ["product_name", "orders"]
    fig = px.bar(
        top_products.sort_values("orders"),
        x="orders",
        y="product_name",
        orientation="h",
        title=f"Top {top_n} Products",
        labels={"product_name": "Product", "orders": "Order lines"},
        color="orders",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        departments = merged["department"].value_counts().head(10).reset_index()
        departments.columns = ["department", "orders"]
        fig = px.bar(
            departments.sort_values("orders"),
            x="orders",
            y="department",
            orientation="h",
            title="Top Departments",
            labels={"department": "Department", "orders": "Order lines"},
            color_discrete_sequence=["#577590"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        aisles = merged["aisle"].value_counts().head(10).reset_index()
        aisles.columns = ["aisle", "orders"]
        fig = px.bar(
            aisles.sort_values("orders"),
            x="orders",
            y="aisle",
            orientation="h",
            title="Top Aisles",
            labels={"aisle": "Aisle", "orders": "Order lines"},
            color_discrete_sequence=["#90be6d"],
        )
        st.plotly_chart(fig, use_container_width=True)

    organic_mask = merged["product_name"].str.contains(
        "Organic", case=organic_case_sensitive, na=False
    )
    organic_counts = pd.Series(
        {
            "Organic": int(organic_mask.sum()),
            "Non-Organic": int((~organic_mask).sum()),
        }
    ).reset_index()
    organic_counts.columns = ["type", "order_lines"]
    fig = px.pie(
        organic_counts,
        names="type",
        values="order_lines",
        title="Organic vs Non-Organic Product Lines",
        color_discrete_sequence=["#4d908e", "#f9c74f"],
        hole=0.45,
    )
    st.plotly_chart(fig, use_container_width=True)


def sample_customer_features(customer_features: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if len(customer_features) <= sample_size:
        return customer_features.copy()
    return customer_features.sample(n=sample_size, random_state=42)


def build_embedding(feature_frame: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_frame[NUMERIC_FEATURES])
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(scaled)
    embedded = feature_frame.copy()
    embedded["pc1"] = embedding[:, 0]
    embedded["pc2"] = embedding[:, 1]
    return embedded, scaled


def fuzzy_cmeans_labels(scaled: np.ndarray, clusters: int, m: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    membership = rng.random((clusters, scaled.shape[0]))
    membership = membership / membership.sum(axis=0, keepdims=True)

    for _ in range(180):
        previous = membership.copy()
        weights = membership**m
        centers = (weights @ scaled) / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        distances = np.linalg.norm(scaled[None, :, :] - centers[:, None, :], axis=2)
        distances = np.maximum(distances, 1e-8)
        inverse_distances = distances ** (-2 / (m - 1))
        membership = inverse_distances / inverse_distances.sum(axis=0, keepdims=True)
        if np.linalg.norm(membership - previous) < 0.005:
            break

    return np.argmax(membership, axis=0), membership.max(axis=0)


def cluster_labels(method: str, scaled: np.ndarray, clusters: int) -> np.ndarray:
    if method == "MiniBatch K-Means":
        model = MiniBatchKMeans(n_clusters=clusters, random_state=42, batch_size=1024, n_init="auto")
        return model.fit_predict(scaled)
    if method == "Fuzzy C-Means":
        labels, _ = fuzzy_cmeans_labels(scaled, clusters)
        return labels
    if method == "Gaussian Mixture":
        model = GaussianMixture(n_components=clusters, random_state=42, covariance_type="full")
        return model.fit_predict(scaled)
    model = AgglomerativeClustering(n_clusters=clusters, linkage="ward")
    return model.fit_predict(scaled)


def evaluate_cluster_range(method: str, scaled: np.ndarray, k_values: Iterable[int]) -> pd.DataFrame:
    rows = []
    for k in k_values:
        labels = cluster_labels(method, scaled, k)
        rows.append(
            {
                "Clusters": k,
                "Silhouette": silhouette_score(scaled, labels),
                "Calinski-Harabasz": calinski_harabasz_score(scaled, labels),
                "Davies-Bouldin": davies_bouldin_score(scaled, labels),
            }
        )
    return pd.DataFrame(rows)


def build_cluster_space(scaled: np.ndarray, representation: str) -> Tuple[np.ndarray, np.ndarray, str, Optional[str]]:
    if representation == "Original standardized":
        pca = PCA(n_components=2, random_state=42)
        plot_space = pca.fit_transform(scaled)
        return scaled, plot_space, "PCA projection of standardized features", None

    if representation == "PCA":
        pca = PCA(n_components=3, random_state=42)
        cluster_space = pca.fit_transform(scaled)
        return cluster_space, cluster_space[:, :2], "PCA representation", None

    if representation == "t-SNE":
        perplexity = max(5, min(30, (scaled.shape[0] - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        cluster_space = tsne.fit_transform(scaled)
        return cluster_space, cluster_space, "t-SNE representation", None

    if representation == "UMAP":
        if importlib.util.find_spec("umap") is None:
            cluster_space, plot_space, label, _ = build_cluster_space(scaled, "PCA")
            return cluster_space, plot_space, label, "UMAP is not installed, so PCA is being used for this run."
        import umap.umap_ as umap

        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        cluster_space = reducer.fit_transform(scaled)
        return cluster_space, cluster_space, "UMAP representation", None

    if representation == "Autoencoder":
        if importlib.util.find_spec("tensorflow") is None:
            cluster_space, plot_space, label, _ = build_cluster_space(scaled, "PCA")
            return cluster_space, plot_space, label, "TensorFlow is not installed, so PCA is being used for this run."
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam

        tf.random.set_seed(42)
        input_layer = Input(shape=(scaled.shape[1],))
        encoded = Dense(8, activation="relu")(input_layer)
        encoded = Dense(2, activation="relu")(encoded)
        decoded = Dense(8, activation="relu")(encoded)
        decoded = Dense(scaled.shape[1], activation="linear")(decoded)
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        encoder = Model(inputs=input_layer, outputs=encoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
        autoencoder.fit(scaled, scaled, epochs=25, batch_size=64, shuffle=True, verbose=0)
        cluster_space = encoder.predict(scaled, verbose=0)
        return cluster_space, cluster_space, "Autoencoder representation", None

    return build_cluster_space(scaled, "Original standardized")


def compare_clustering_methods(cluster_space: np.ndarray, clusters: int) -> pd.DataFrame:
    rows = []
    for method in CLUSTER_METHODS:
        labels = cluster_labels(method, cluster_space, clusters)
        rows.append(
            {
                "Method": method,
                "Clusters": clusters,
                "Silhouette": silhouette_score(cluster_space, labels),
                "Calinski-Harabasz": calinski_harabasz_score(cluster_space, labels),
                "Davies-Bouldin": davies_bouldin_score(cluster_space, labels),
            }
        )
    return pd.DataFrame(rows)


def describe_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    overall = clustered[NUMERIC_FEATURES].mean()
    profiles = clustered.groupby("cluster")[NUMERIC_FEATURES].mean()
    profiles["customers"] = clustered["cluster"].value_counts().sort_index()

    descriptions = []
    for _, row in profiles.iterrows():
        labels = []
        if row["total_orders"] > overall["total_orders"] and row["average_days_between_orders"] < overall["average_days_between_orders"]:
            labels.append("High-engagement shoppers")
        if row["reorder_rate"] > overall["reorder_rate"]:
            labels.append("Retention customers")
        if row["unique_products"] > overall["unique_products"] and row["reorder_rate"] < overall["reorder_rate"]:
            labels.append("Variety seekers")
        if row["average_days_between_orders"] > overall["average_days_between_orders"] * 1.15:
            labels.append("At-risk shoppers")
        descriptions.append(", ".join(labels) if labels else "Steady shoppers")
    profiles["description"] = descriptions
    return profiles.reset_index()


def render_segmentation(customer_features: pd.DataFrame) -> None:
    st.caption("Clustering is run on customer-level behavior features with the same evaluation metrics used in the thesis.")
    controls = st.columns(5)
    with controls[0]:
        method = st.selectbox(
            "Clustering method",
            CLUSTER_METHODS,
        )
    with controls[1]:
        representation = st.selectbox("Representation", REPRESENTATIONS)
    with controls[2]:
        clusters = st.slider("Clusters", 2, 9, 4)
    with controls[3]:
        sample_cap = 10_000
        if method in {"Agglomerative", "Fuzzy C-Means"}:
            sample_cap = min(sample_cap, 5_000)
        if representation in {"t-SNE", "UMAP", "Autoencoder"}:
            sample_cap = min(sample_cap, 3_000)
        max_sample = min(10_000, len(customer_features))
        max_sample = min(sample_cap, max_sample)
        min_sample = min(100, max_sample)
        step = 100 if max_sample >= 500 else 25
        sample_size = st.slider(
            "Customer sample",
            min_sample,
            max_sample,
            min(3_000, max_sample),
            step=step,
        )
    with controls[4]:
        run_range = st.toggle("Evaluate k=2..9", value=True)

    sampled = sample_customer_features(customer_features.dropna(subset=NUMERIC_FEATURES), sample_size)
    embedded, scaled = build_embedding(sampled)
    cluster_space, plot_space, axis_label, fallback_note = build_cluster_space(scaled, representation)
    if fallback_note:
        st.info(fallback_note)

    labels = cluster_labels(method, cluster_space, clusters)
    embedded["cluster"] = labels.astype(str)
    embedded["x_axis"] = plot_space[:, 0]
    embedded["y_axis"] = plot_space[:, 1]
    if method == "Fuzzy C-Means":
        _, confidence = fuzzy_cmeans_labels(cluster_space, clusters)
        embedded["membership_strength"] = confidence

    left, right = st.columns((1.15, 1))
    with left:
        fig = px.scatter(
            embedded,
            x="x_axis",
            y="y_axis",
            color="cluster",
            hover_data=[
                "user_id",
                "top_department",
                "top_aisle",
                "reorder_rate",
                "total_orders",
                "membership_strength" if method == "Fuzzy C-Means" else "average_basket_size",
            ],
            title=f"{method} Customer Segments",
            labels={"x_axis": f"{axis_label} 1", "y_axis": f"{axis_label} 2"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        cluster_profile = describe_clusters(embedded.assign(cluster=labels))
        st.dataframe(
            cluster_profile[
                [
                    "cluster",
                    "customers",
                    "description",
                    "total_orders",
                    "average_days_between_orders",
                    "reorder_rate",
                    "unique_products",
                    "average_basket_size",
                ]
            ].round(3),
            use_container_width=True,
            hide_index=True,
        )

    if run_range:
        with st.spinner("Evaluating cluster counts..."):
            scores = evaluate_cluster_range(method, cluster_space, range(2, 10))
        score_fig = go.Figure()
        score_fig.add_trace(
            go.Scatter(x=scores["Clusters"], y=scores["Silhouette"], mode="lines+markers", name="Silhouette")
        )
        score_fig.add_trace(
            go.Scatter(
                x=scores["Clusters"],
                y=scores["Davies-Bouldin"],
                mode="lines+markers",
                name="Davies-Bouldin",
                yaxis="y2",
            )
        )
        score_fig.update_layout(
            title="Cluster Quality by k",
            xaxis_title="Clusters",
            yaxis_title="Silhouette",
            yaxis2=dict(title="Davies-Bouldin", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(score_fig, use_container_width=True)
        st.dataframe(scores.round(4), use_container_width=True, hide_index=True)

    compare = st.toggle("Compare all clustering methods at selected k", value=False)
    if compare:
        with st.spinner("Comparing clustering methods..."):
            method_scores = compare_clustering_methods(cluster_space, clusters)
        fig = px.bar(
            method_scores.sort_values("Silhouette"),
            x="Silhouette",
            y="Method",
            orientation="h",
            title=f"Method Benchmark on {axis_label}",
            color="Davies-Bouldin",
            color_continuous_scale="Tealrose",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(method_scores.round(4), use_container_width=True, hide_index=True)


def add_churn_label(customer_features: pd.DataFrame, orders: pd.DataFrame, threshold_days: int) -> pd.DataFrame:
    latest_orders = (
        orders.sort_values(["user_id", "order_number"])
        .drop_duplicates("user_id", keep="last")
        [["user_id", "days_since_prior_order"]]
        .rename(columns={"days_since_prior_order": "days_since_last_order"})
    )
    labeled = customer_features.merge(latest_orders, on="user_id", how="left")
    labeled["days_since_last_order"] = labeled["days_since_last_order"].fillna(0)
    labeled["churn"] = (labeled["days_since_last_order"] > threshold_days).astype(int)
    return labeled


def render_churn_model(customer_features: pd.DataFrame, orders: pd.DataFrame) -> None:
    st.caption(
        "Churn is labeled as customers whose latest observed gap since prior order exceeds the selected day threshold."
    )
    threshold = st.slider("Churn threshold days", 10, 120, 30, step=5)
    labeled = add_churn_label(customer_features, orders, threshold)
    churn_rate = labeled["churn"].mean()
    st.metric("Labeled churn rate", f"{churn_rate:.1%}")

    if labeled["churn"].nunique() < 2:
        st.info("The current threshold creates only one class. Move the threshold to train the model.")
        return

    X = labeled[NUMERIC_FEATURES].fillna(0)
    y = labeled["churn"]
    test_size = 0.25 if len(labeled) >= 1_000 else 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=test_size,
        random_state=42,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_name = "Random Forest"
    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        neg, pos = np.bincount(y_train)
        scale_pos_weight = neg / max(pos, 1)
        model = XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_estimators=180,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
        )
        model_name = "XGBoost"
    else:
        model = RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=4,
            n_jobs=-1,
        )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    left, right = st.columns(2)
    with left:
        st.metric("Model", model_name)
        st.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
        st.metric("PR AUC", f"{average_precision_score(y_test, y_proba):.3f}")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    with right:
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Churn", "Churn"],
            y=["Not Churn", "Churn"],
            title="Confusion Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=model_name))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
    with curve_col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=model_name))
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    importances = pd.DataFrame(
        {"feature": NUMERIC_FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    fig = px.bar(
        importances,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance",
        labels={"feature": "Feature", "importance": "Importance"},
        color_discrete_sequence=["#f8961e"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def render_data_quality(frames: Dict[str, pd.DataFrame], customer_features: pd.DataFrame) -> None:
    rows = []
    for name, frame in frames.items():
        rows.append(
            {
                "table": CSV_FILES[name],
                "rows": len(frame),
                "columns": len(frame.columns),
                "duplicate_rows": int(frame.duplicated().sum()),
                "missing_values": int(frame.isna().sum().sum()),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Customer Feature Preview")
    st.dataframe(customer_features.head(200), use_container_width=True, hide_index=True)

    csv = customer_features.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download customer features",
        data=csv,
        file_name="instacart_customer_features.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Instacart Customer Segmentation Dashboard")
    st.caption("Interactive EDA, product behavior, clustering, and churn modeling for the Instacart market basket dataset.")

    with st.sidebar:
        st.header("Data")
        backend_data_dir = find_backend_data_dir()
        if backend_data_dir:
            st.success(f"Backend dataset detected: `{backend_data_dir}`")
        else:
            st.warning("Backend dataset not found. Upload CSVs or enable demo data.")
        uploaded = st.file_uploader(
            "Optional CSV override",
            type="csv",
            accept_multiple_files=True,
            help="Expected files: orders.csv, order_products__prior.csv, order_products__train.csv, products.csv, aisles.csv, departments.csv",
        )
        use_demo = st.toggle("Use demo data when CSVs are missing", value=True)
        order_source = st.selectbox(
            "Order-product source",
            ["Prior + train orders", "Prior orders only", "Train orders only"],
        )
        st.divider()
        st.markdown("Place full dataset CSVs in `data/`, or set `INSTACART_DATA_DIR`.")

    if uploaded:
        frames = load_uploaded_data(uploaded)
    else:
        frames = load_local_data()

    missing = [name for name, frame in frames.items() if frame is None]
    if missing and use_demo:
        st.info("Using generated demo data because one or more Instacart CSVs were not found.")
        frames = make_demo_data()
    elif missing:
        missing_data_message(missing)
        st.stop()

    complete_frames = {name: frame for name, frame in frames.items() if frame is not None}
    orders, merged, customer_features = prepare_analysis(complete_frames, order_source)

    overview_tab, products_tab, segments_tab, churn_tab, data_tab = st.tabs(
        ["Overview", "Products", "Segments", "Churn Model", "Data Quality"]
    )
    with overview_tab:
        render_overview(orders, merged, customer_features)
    with products_tab:
        render_product_views(merged)
    with segments_tab:
        render_segmentation(customer_features)
    with churn_tab:
        render_churn_model(customer_features, orders)
    with data_tab:
        render_data_quality(complete_frames, customer_features)


if __name__ == "__main__":
    main()
