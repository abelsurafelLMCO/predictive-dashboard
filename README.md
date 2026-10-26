# Predictive‑Analytics Dashboard (One‑Click Deploy)

A fully‑featured interactive dashboard built with **Dash / Plotly** that:

* Shows a data table & summary statistics.
* Forecasts the next 30 days of the target variable.
* Visualises global feature importance with SHAP.
* Lets you experiment with **What‑If** scenarios in real time.
* Provides model‑performance metrics and download links for the model & predictions.

## How to launch (no code on your machine)

1. **Create a GitHub repo** (or ask a colleague to do it) and push the whole `predictive-dashboard/` folder.
2. **Connect the repo to a free hosting service** – we recommend **Streamlit Community Cloud** because it requires only a GitHub URL and a single button click.

   * Go to https://share.streamlit.io/
   * Click **“New app”**, select the repository and branch (`main`), and set the *app file path* to `app.py`.
   * Click **“Deploy”**. After a few seconds you’ll receive a public URL like `https://your‑username‑predictive‑dashboard.streamlit.app`.

3. 🎉 The dashboard is live! Share the URL with anyone; they don’t need a Python environment.

### Updating the data

If you ever need to replace `data/your_data.csv`:

1. Replace the file in the repo (keep the same name).
2. Commit & push.
3. The hosting platform automatically rebuilds and redeploys (usually < 2 min).

### Running locally (optional)

```bash
git clone https://github.com/your-username/predictive-dashboard.git
cd predictive-dashboard
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
