# Customer Segmentation using Instacart Dataset

This project analyses Instacart customer behavior using the CRISP-DM data mining methodology. The research uses exploratory data analysis, data cleaning, feature engineering, data visualization, clustering, dimensionality reduction, cluster profiling, and churn prediction to study consumer behavior in a grocery retail context.

The original analysis works with the Instacart dataset of over 3.4 million orders. It compares clustering methods including K-Means, Fuzzy C-Means, Agglomerative Clustering, and Gaussian Mixture Models, alongside dimensionality-reduction techniques such as PCA, t-SNE, UMAP, and Autoencoders. Cluster counts are evaluated with Silhouette score, Davies-Bouldin Index, and Calinski-Harabasz Index. The project also models likely churn using a 30-day ordering gap and XGBoost.

## Dashboard

This repository turns the notebook analysis into a Streamlit dashboard with:

- dataset health checks and missing-value summaries
- order behavior visualizations
- top products, departments, aisles, and organic share
- customer feature engineering
- K-Means, Fuzzy C-Means, Gaussian Mixture, and Agglomerative clustering
- PCA, t-SNE, UMAP, and optional Autoencoder representations
- segment visualizations, cluster profiles, and clustering metric benchmarks
- churn modeling using XGBoost when available, with a Random Forest fallback

## Files

- `app.py`: Streamlit dashboard
- `Customer_segmentation_instacart.py`: original Python analysis exported from the notebook
- `requirements.txt`: Python dependencies
- `Dockerfile`: container setup for deployment
- `render.yaml`: Render Blueprint configuration

## Dataset

Download the Instacart Market Basket Analysis CSV files and put them in a `data/` folder:

```text
data/
  orders.csv
  order_products__prior.csv
  order_products__train.csv
  products.csv
  aisles.csv
  departments.csv
```

The full CSV files are intentionally excluded from git because they are large. The app can also run with generated demo data when the CSV files are missing, so the deployed dashboard remains testable.

For local development on the original machine, the app also checks:

```text
/Users/aarushi/Desktop/Dissertation/5663439_Copy
```

If your deployment mounts the dataset somewhere else, set:

```bash
INSTACART_DATA_DIR=/path/to/instacart/csvs
```

## Run Locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL printed in the terminal.

TensorFlow is optional. If it is installed, the Autoencoder representation will be available; otherwise the dashboard falls back to PCA for that selection.

## Deploy on Render

The repository includes a `Dockerfile` and `render.yaml` so the dashboard can be hosted as a Render web service.

1. Push this repository to GitHub.
2. In Render, create a new Blueprint or Web Service from the repository.
3. Use Docker as the runtime. Render will use the included `Dockerfile`.
4. The included `render.yaml` uses the free plan to avoid accidental billing.
5. Keep the entrypoint as:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

For a public hosted demo, leave demo data enabled. For a full-data deployment, attach provider storage and mount the CSV files under `data/`, or set `INSTACART_DATA_DIR`.

## Can this run on GitHub Pages?

Not as a Streamlit app. GitHub Pages hosts static files such as HTML, CSS, and JavaScript. This dashboard is a Python web app that needs a running Streamlit server.

## Docker

Build and run locally:

```bash
docker build -t instacart-dashboard .
docker run --rm -p 8501:8501 instacart-dashboard
```
