import json
import pathlib
import datetime as dt

import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate

# ----------------------------------------------------------------------
# Load pre‑trained artifacts (they are static and ship with the repo)
ARTIFACTS = pathlib.Path("model_artifacts")
preprocess = joblib.load(ARTIFACTS / "preprocessor.joblib")
model = joblib.load(ARTIFACTS / "model.joblib")
explainer = joblib.load(ARTIFACTS / "explainer.joblib")

with open(ARTIFACTS / "metrics.json") as f:
    METRICS = json.load(f)
with open(ARTIFACTS / "columns.json") as f:
    COL_ORDER = json.load(f)

# ----------------------------------------------------------------------
# Helper: turn a raw dict into the vector expected by the model
def row_to_features(row: dict) -> np.ndarray:
    """Create a 1‑row DataFrame that matches training schema then preprocess."""
    df = pd.DataFrame([row])

    # Date → expanded features (if present)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["year"]      = df["date"].dt.year
        df["month"]     = df["date"].dt.month
        df["day"]       = df["date"].dt.day
        df["weekday"]   = df["date"].dt.weekday
        df["dayofyear"] = df["date"].dt.dayofyear
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)
        df = df.drop(columns=["date"])

    return preprocess.transform(df)

# ----------------------------------------------------------------------
# Initialize Dash
app = Dash(__name__, title="Predictive Analytics Dashboard", suppress_callback_exceptions=True)
server = app.server  # needed for Render/Fly.io

# ----------------------------------------------------------------------
# Layout (Bootstrap‑styled for a clean look)
app.layout = html.Div(
    style={"fontFamily": "system-ui", "margin": "20px"},
    children=[
        html.H1("📊 Predictive‑Analytics Dashboard", style={"textAlign": "center"}),
        html.Hr(),

        # --------------------------------------------------------------
        # 1️⃣ Data Overview
        html.Div([
            html.H2("🗂️ Data Overview"),
            dcc.Dropdown(
                id="cols-toggle",
                placeholder="Select columns to display (default = all)",
                multi=True,
                style={"width": "60%"}
            ),
            dash_table.DataTable(
                id="data-table",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "5px"},
                style_header={"backgroundColor": "#f7f7f7", "fontWeight": "bold"},
            ),
        ], style={"marginBottom": "40px"}),

        # --------------------------------------------------------------
        # 2️⃣ Forecast
        html.Div([
            html.H2("⏱️ 30‑Day Forecast"),
            dcc.DatePickerRange(
                id="forecast-range",
                start_date=dt.date.today(),
                end_date=dt.date.today() + dt.timedelta(days=30),
                display_format="YYYY‑MM‑DD"
            ),
            dcc.Graph(id="forecast-graph")
        ], style={"marginBottom": "40px"}),

        # --------------------------------------------------------------
        # 3️⃣ SHAP Summary
        html.Div([
            html.H2("🔎 Global Feature Impact (SHAP)"),
            dcc.Graph(id="shap-summary")
        ], style={"marginBottom": "40px"}),

        # --------------------------------------------------------------
        # 4️⃣ What‑If Scenario
        html.Div([
            html.H2("🧪 What‑If Scenario"),
            html.Div(id="whatif-controls", style={"columnCount": 2}),
            html.Button("🔮 Predict", id="whatif-predict", n_clicks=0,
                        style={"marginTop": "10px"}),
            html.Div(id="whatif-output", style={"marginTop": "20px", "fontSize": "1.2em"}),
        ], style={"marginBottom": "40px"}),

        # --------------------------------------------------------------
        # 5️⃣ Model Metrics
        html.Div([
            html.H2("📈 Model Performance"),
            html.Ul([
                html.Li(f"Mean Absolute Error (MAE): {METRICS['mae']:.2f}"),
                html.Li(f"Root Mean Squared Error (RMSE): {METRICS['rmse']:.2f}"),
                html.Li(f"R² Score: {METRICS['r2']:.4f}")
            ])
        ]),

        # --------------------------------------------------------------
        # 6️⃣ Download Links
        html.Div([
            html.H2("💾 Download"),
            html.A("Model (joblib)", href="/download/model", style={"marginRight": "20px"}),
            html.A("Latest 30‑day predictions (CSV)", href="/download/predictions")
        ], style={"marginBottom": "40px"}),
    ]
)

# ----------------------------------------------------------------------
# Callback – populate column dropdown (once on page load)
@app.callback(
    Output("cols-toggle", "options"),
    Output("cols-toggle", "value"),
    Input("data-table", "data")  # dummy trigger
)
def populate_column_dropdown(_):
    df = pd.read_csv("data/your_data.csv")
    opts = [{"label": c, "value": c} for c in df.columns]
    return opts, []  # empty list => show all columns by default

# ----------------------------------------------------------------------
# Callback – update data table based on selected columns
@app.callback(
    Output("data-table", "data"),
    Output("data-table", "columns"),
    Input("cols-toggle", "value")
)
def update_table(selected_cols):
    df = pd.read_csv("data/your_data.csv")
    if selected_cols:
        df = df[selected_cols]
    columns = [{"name": c, "id": c} for c in df.columns]
    return df.to_dict("records"), columns

# ----------------------------------------------------------------------
# Callback – Forecast graph (simple forward propagation)
@app.callback(
    Output("forecast-graph", "figure"),
    Input("forecast-range", "start_date"),
    Input("forecast-range", "end_date")
)
def plot_forecast(start, end):
    if not start or not end:
        raise PreventUpdate

    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
    future_dates = pd.date_range(start_dt, end_dt, freq="D")

    # Build a "most recent row" template and replace the date
    base_row = pd.read_csv("data/your_data.csv").iloc[-1].to_dict()
    rows = []
    for d in future_dates:
        r = base_row.copy()
        r["date"] = d.strftime("%Y-%m-%d")
        rows.append(r)

    X_future = pd.DataFrame(rows)
    X_future_pp = preprocess.transform(X_future)
    preds = model.predict(X_future_pp)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=preds,
                             mode="lines+markers", name="Predicted"))
    fig.update_layout(
        title="30‑Day Target Forecast",
        xaxis_title="Date",
        yaxis_title="Target (predicted)",
        template="plotly_white"
    )
    return fig

# ----------------------------------------------------------------------
# Callback – SHAP Summary (static, computed on a sample)
@app.callback(
    Output("shap-summary", "figure"),
    Input("data-table", "data")  # dummy trigger
)
def render_shap(_):
    # Sample a subset for speed
    df = pd.read_csv("data/your_data.csv").sample(300, random_state=42)
    df_pp = preprocess.transform(df)
    shap_vals = explainer.shap_values(df_pp)

    # Plotly version of SHAP summary
    # -------------------------------------------------
    # Convert SHAP matrix to DataFrame with column names
    feature_names = COL_ORDER
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    # For each feature compute mean absolute SHAP -> sort
    mean_shap = shap_df.abs().mean().sort_values(ascending=False)
    top_features = mean_shap.head(15).index.tolist()

    # Build a "beeswarm" style plot manually (more compact than shap's native)
    fig = go.Figure()
    for i, feat in enumerate(top_features):
        y = np.random.randn(shap_df.shape[0]) * 0.1 + i   # jitter
        fig.add_trace(go.Scatter(
            x=shap_df[feat],
            y=y,
            mode="markers",
            marker=dict(size=5, opacity=0.6),
            name=feat,
            hovertemplate=f"{feat}= %{{x:.3f}}"
        ))
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title="SHAP Summary (Top 15 features)",
        xaxis_title="SHAP value",
        yaxis=dict(showgrid=False, zeroline=False),
        height=600,
        template="plotly_white"
    )
    return fig

# ----------------------------------------------------------------------
# Callback – Dynamically generate What‑If controls
@app.callback(
    Output("whatif-controls", "children"),
    Input("data-table", "data")  # dummy trigger
)
def build_whatif_controls(_):
    df = pd.read_csv("data/your_data.csv")
    controls = []
    for col in df.columns:
        if col == "target":
            continue
        dtype = df[col].dtype
        # Date picker
        if col == "date":
            controls.append(html.Label("Date:", className="my-1"))
            controls.append(
                dcc.DatePickerSingle(
                    id={"type": "whatif", "index": col},
                    date=dt.date.today()
                )
            )
        # Categorical dropdown
        elif dtype == "object" or str(dtype).startswith("category"):
            opts = [{"label": v, "value": v} for v in sorted(df[col].unique())]
            controls.append(html.Label(f"{col.title()}:", className="my-1"))
            controls.append(
                dcc.Dropdown(
                    id={"type": "whatif", "index": col},
                    options=opts,
                    value=opts[0]["value"]
                )
            )
        # Binary/boolean (0‑1)
        elif set(df[col].unique()) <= {0, 1}:
            controls.append(html.Label(f"{col.title()}:", className="my-1"))
            controls.append(
                dcc.RadioItems(
                    id={"type": "whatif", "index": col},
                    options=[
                        {"label": "No", "value": 0},
                        {"label": "Yes", "value": 1}
                    ],
                    value=0,
                    labelStyle={"display": "inline-block", "marginRight": "10px"}
                )
            )
        # Numeric input
        else:
            min_val, max_val = df[col].min(), df[col].max()
            mean_val = df[col].mean()
            controls.append(html.Label(f"{col.title()}:", className="my-1"))
            controls.append(
                dcc.Input(
                    id={"type": "whatif", "index": col},
                    type="number",
                    value=round(mean_val, 2),
                    min=min_val,
                    max=max_val,
                    step=0.01,
                    style={"width": "100%"}
                )
            )
    return controls

# ----------------------------------------------------------------------
# Callback – run what‑if prediction
@app.callback(
    Output("whatif-output", "children"),
    Input("whatif-predict", "n_clicks"),
    State({"type": "whatif", "index": ALL}, "value"),
    State({"type": "whatif", "index": ALL}, "id")
)
def run_whatif(n_clicks, values, ids):
    if n_clicks == 0:
        raise PreventUpdate
    # Build dict from ids/value pairs
    row = {item["index"]: val for item, val in zip(ids, values)}
    X_vec = row_to_features(row)
    pred = model.predict(X_vec)[0]
    return html.Div([
        html.Strong("Predicted target: "),
        f"{pred:,.2f}"
    ])

# ----------------------------------------------------------------------
# Download routes (Flask‑style)
@server.route("/download/model")
def download_model():
    from flask import send_from_directory
    return send_from_directory("model_artifacts", "model.joblib", as_attachment=True)

@server.route("/download/predictions")
def download_predictions():
    # Re‑run the 30‑day forecast and return CSV
    start_dt = pd.to_datetime(dt.date.today())
    dates = pd.date_range(start_dt, periods=30)
    base_row = pd.read_csv("data/your_data.csv").iloc[-1].to_dict()
    rows = []
    for d in dates:
        r = base_row.copy()
        r["date"] = d.strftime("%Y-%m-%d")
        rows.append(r)
    X_fut = pd.DataFrame(rows)
    preds = model.predict(preprocess.transform(X_fut))
    out = pd.DataFrame({"date": dates, "predicted_target": preds})
    out_path = "model_artifacts/predictions.csv"
    out.to_csv(out_path, index=False)
    from flask import send_from_directory
    return send_from_directory("model_artifacts", "predictions.csv", as_attachment=True)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # For local testing
    app.run_server(debug=False, host="0.0.0.0", port=8050)
