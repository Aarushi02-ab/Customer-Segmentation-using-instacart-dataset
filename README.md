# Instacart Customer Segmentation Dashboard

This project turns the Instacart notebook analysis into a Streamlit dashboard with:

- dataset health checks and missing-value summaries
- order behavior visualizations
- top products, departments, aisles, and organic share
- customer feature engineering
- K-Means, Gaussian Mixture, and Agglomerative clustering
- Fuzzy C-Means clustering implemented directly in NumPy
- PCA, t-SNE, UMAP, and optional Autoencoder representations
- segment visualizations, cluster profiles, and clustering metric benchmarks
- churn modeling using XGBoost when available, with a Random Forest fallback

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

The app also looks in the dissertation data folder used for this project:

```text
/Users/aarushi/Desktop/Dissertation/5663439_Copy
```

You can also upload the same CSVs from the app sidebar. If the files are missing, the app can run with generated demo data so the dashboard layout is still testable.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL printed in the terminal.

TensorFlow is optional. If it is installed, the Autoencoder representation will be available; otherwise the dashboard falls back to PCA for that selection.

## Deploy Permanently

The repository includes a `Dockerfile` and `render.yaml` so the dashboard can be hosted as a persistent web service.

### Can this run on GitHub Pages?

Not as a Streamlit app. GitHub Pages hosts static files such as HTML, CSS, and JavaScript. This dashboard is a Python web app that needs a running Streamlit server, so GitHub Pages cannot execute it.

The closest GitHub-based deployment is:

1. Push this repository to GitHub.
2. Deploy the app from that GitHub repository on Streamlit Community Cloud.
3. Use `app.py` as the entrypoint file.
4. Keep `requirements.txt` in the repository root.

Streamlit Community Cloud will run the Python app and provide a public `streamlit.app` URL.

### Option 1: Render

1. Push this project to GitHub.
2. In Render, create a new Blueprint or Web Service from the repository.
3. Use Docker as the runtime. Render will use the included `Dockerfile`.
4. The included `render.yaml` uses the free plan to avoid accidental billing. For an always-on deployment, change `plan: free` to a paid web-service instance such as `plan: starter`.
5. Keep the entrypoint as:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

Render can redeploy automatically whenever you push changes to the linked branch.

### Option 2: Any Docker Cloud Host

Build and run the container locally:

```bash
docker build -t instacart-dashboard .
docker run --rm -p 8501:8501 instacart-dashboard
```

On a cloud VM or container platform, expose the service port and set `PORT` if the provider requires a specific value.

### Dataset on Cloud

The full Instacart CSV files are intentionally excluded from git because they are large. For a public hosted demo, leave demo data enabled. For a full-data deployment, upload the CSVs through the sidebar each session or attach provider storage and mount the CSV files under `data/`.

If your cloud provider mounts the dataset somewhere else, set:

```bash
INSTACART_DATA_DIR=/path/to/instacart/csvs
```
