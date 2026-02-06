
"""
Streamlit Wine Quality Classification App (OOP, production-style)

This app:
1) Loads a user-provided CSV (red wine dataset format; ';' delimiter).
2) Binarizes quality into {0,1} using a slider threshold.
3) Allows optional column dropping + renaming for friendly labels.
4) Trains/evaluates multiple models with consistent pipelines.
5) Visualizes correlations, confusion matrix, and ROC curve (where applicable).

Expected CSV columns (typical UCI wine quality dataset):
- fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
  free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, quality
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# =========================
# Config / Defaults
# =========================

FRIENDLY_RENAMES = {
    "fixed acidity": "Sourness",
    "citric acid": "Fruitiness",
    "residual sugar": "Sweetness",
    "chlorides": "Saltiness",
    "total sulfur dioxide": "Preservatives",
    "alcohol": "Alcohol%",
    "quality": "Quality",
}

DEFAULT_DROP_COLS = [
    "volatile acidity",
    "free sulfur dioxide",
    "pH",
    "sulphates",
    "density",
]


@dataclass(frozen=True)
class AppConfig:
    """Configuration values controlling training/evaluation behavior."""
    test_size: float = 0.25


# =========================
# Data layer
# =========================

class WineDataLoader:
    """Loads and validates the wine dataset from an uploaded file-like object."""

    REQUIRED_TARGET_COL = "quality"

    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_csv(uploaded_file, sep: str = ";") -> pd.DataFrame:
        """
        Load CSV from Streamlit UploadedFile.

        Parameters
        ----------
        uploaded_file:
            Streamlit UploadedFile.
        sep:
            CSV separator; UCI wine dataset uses ';'.

        Returns
        -------
        pd.DataFrame
        """
        df = pd.read_csv(uploaded_file, sep=sep)
        if WineDataLoader.REQUIRED_TARGET_COL not in df.columns:
            raise ValueError(
                f"CSV must include '{WineDataLoader.REQUIRED_TARGET_COL}' column."
            )
        return df


class WinePreprocessor:
    """Transforms raw dataset into modeling-ready dataframe."""

    def __init__(self, score_limit: int, drop_cols: Optional[List[str]] = None, use_friendly_names: bool = True):
        self.score_limit = int(score_limit)
        self.drop_cols = drop_cols or []
        self.use_friendly_names = use_friendly_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Binarize quality and apply column drops and optional renames.

        Returns a new dataframe (does not mutate input).
        """
        out = df.copy()

        # Binarize
        out["quality"] = np.where(out["quality"] >= self.score_limit, 1, 0)

        # Drop columns if present
        cols_to_drop = [c for c in self.drop_cols if c in out.columns]
        if cols_to_drop:
            out = out.drop(columns=cols_to_drop)

        # Friendly renames
        if self.use_friendly_names:
            rename_map = {k: v for k, v in FRIENDLY_RENAMES.items() if k in out.columns}
            out = out.rename(columns=rename_map)

        return out


# =========================
# Modeling layer
# =========================

class ModelFactory:
    """Creates sklearn models based on user choice."""

    @staticmethod
    def make_models(random_seed: int) -> Dict[str, Pipeline]:
        """
        Return a dictionary of named model pipelines.
        (Pipelines are ready for future feature engineering steps.)
        """
        # If you add scaling/encoding later, put it inside the Pipeline.
        return {
            "Decision Tree (depth=5)": Pipeline(
                steps=[("model", DecisionTreeClassifier(random_state=random_seed, max_depth=5))]
            ),
            "Decision Tree (depth=10)": Pipeline(
                steps=[("model", DecisionTreeClassifier(random_state=random_seed, max_depth=10))]
            ),
            "Random Forest (n=250)": Pipeline(
                steps=[("model", RandomForestClassifier(random_state=random_seed, n_estimators=250))]
            ),
        }


@dataclass
class ModelResult:
    name: str
    accuracy: float
    f1: float
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    feature_names: List[str]


class Trainer:
    """Train/evaluate helper."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def split(
        self,
        df: pd.DataFrame,
        target_col: str,
        features: List[str],
        random_seed: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df[features]
        y = df[target_col]
        return train_test_split(X, y, test_size=self.cfg.test_size, random_state=random_seed, stratify=y)

    def fit_eval(
        self,
        model_name: str,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        feature_names: List[str],
    ) -> ModelResult:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_val)[:, 1]
            except Exception:
                y_proba = None

        return ModelResult(
            name=model_name,
            accuracy=float(accuracy_score(y_val, y_pred)),
            f1=float(f1_score(y_val, y_pred)),
            y_true=y_val.to_numpy(),
            y_pred=np.asarray(y_pred),
            y_proba=y_proba,
            feature_names=feature_names,
        )


# =========================
# Visualization / UI helpers
# =========================

class Viz:
    """Plotting utilities returning matplotlib figures for Streamlit."""

    @staticmethod
    def correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            corr, mask=mask, ax=ax, center=0,
            linewidths=0.5, cbar_kws={"shrink": 0.7}
        )
        ax.set_title("Correlation heatmap (numeric features)")
        return fig

    @staticmethod
    def confusion_matrix_fig(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title("Confusion Matrix")
        return fig

    @staticmethod
    def roc_curve_fig(y_true: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
        ax.set_title("ROC Curve")
        return fig


# =========================
# Streamlit App
# =========================

class WineApp:
    """Main Streamlit application controller."""

    def __init__(self):
        self.cfg = AppConfig()
        self.trainer = Trainer(self.cfg)

    def run(self) -> None:
        st.set_page_config(
            page_title="Ex-stream-ly Cool Wine App",
            page_icon="🍷",
            layout="wide",
            initial_sidebar_state="auto",
        )

        st.title("De-code Red Wines 🍷🍷")
        st.subheader("AI experiment to determine wine quality")

        # Sidebar controls
        with st.sidebar:
            st.header("Controls")
            score_limit = st.slider("Wine Score Limit (>= is 'good')", min_value=0, max_value=10, value=4)
            random_seed = st.slider("Random Seed", min_value=0, max_value=1000, value=123)

        # Step 1: Upload
        st.markdown("### Step 1: Load a CSV file")
        uploaded_file = st.file_uploader("Upload winequality-red CSV", type=["csv"])

        if uploaded_file is None:
            st.info("Upload a CSV to begin.")
            return

        try:
            raw_df = WineDataLoader.load_csv(uploaded_file, sep=";")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            return

        st.success("Data loaded.")
        with st.expander("Preview raw data"):
            st.dataframe(raw_df.head(20), use_container_width=True)

        # Step 2: Drop columns
        st.markdown("### Step 2: Drop non-relevant columns (optional)")
        available_cols = list(raw_df.columns)

        default_drops = [c for c in DEFAULT_DROP_COLS if c in available_cols]
        drop_cols = st.multiselect("Columns to drop", options=available_cols, default=default_drops)

        # Step 3: Preprocess
        pre = WinePreprocessor(score_limit=score_limit, drop_cols=drop_cols, use_friendly_names=True)
        df = pre.transform(raw_df)

        st.markdown("### Step 3: Describe current data")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("Shape:", df.shape)
            st.dataframe(df.describe().T, use_container_width=True)
        with c2:
            st.write("Target balance (Quality)")
            st.bar_chart(df["Quality"].value_counts() if "Quality" in df.columns else df["quality"].value_counts())

        # Identify target col after rename
        target_col = "Quality" if "Quality" in df.columns else "quality"

        # Step 4: Explore correlations
        st.markdown("### Step 4: Explore correlations")
        with st.expander("Correlation heatmap"):
            fig = Viz.correlation_heatmap(df.drop(columns=[target_col]))
            st.pyplot(fig, clear_figure=True)

        # Step 5: Modeling
        st.markdown("## Step 5: Train classification models")

        # Feature selection
        feature_candidates = [c for c in df.columns if c != target_col]
        default_features = [c for c in ["Sourness", "Fruitiness", "Sweetness"] if c in feature_candidates]
        if not default_features:
            default_features = feature_candidates[:3]  # fallback

        features = st.multiselect("Select features", options=feature_candidates, default=default_features)

        if len(features) < 1:
            st.warning("Select at least one feature to train.")
            return

        X_train, X_val, y_train, y_val = self.trainer.split(
            df=df, target_col=target_col, features=features, random_seed=random_seed
        )

        models = ModelFactory.make_models(random_seed=random_seed)

        tab_names = ["Results", *models.keys()]
        tabs = st.tabs(tab_names)

        results: List[ModelResult] = []

        # Train all models once
        for name, pipe in models.items():
            res = self.trainer.fit_eval(
                model_name=name,
                pipeline=pipe,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                feature_names=features,
            )
            results.append(res)

        # Results tab
        with tabs[0]:
            st.subheader("Model comparison")
            summary = pd.DataFrame(
                [{
                    "Model": r.name,
                    "Accuracy (%)": round(r.accuracy * 100, 2),
                    "F1 (%)": round(r.f1 * 100, 2),
                    "Features": ", ".join(r.feature_names),
                } for r in results]
            ).sort_values(by="Accuracy (%)", ascending=False)

            st.dataframe(summary, use_container_width=True)

        # Individual model tabs
        for i, r in enumerate(results, start=1):
            with tabs[i]:
                st.subheader(r.name)
                st.write(f"**Accuracy:** {r.accuracy:.3f}")
                st.write(f"**F1:** {r.f1:.3f}")

                c1, c2 = st.columns(2)

                with c1:
                    st.pyplot(Viz.confusion_matrix_fig(r.y_true, r.y_pred), clear_figure=True)

                with c2:
                    if r.y_proba is not None:
                        st.pyplot(Viz.roc_curve_fig(r.y_true, r.y_proba), clear_figure=True)
                    else:
                        st.info("ROC curve not available (model has no probability output).")

                with st.expander("Validation predictions (sample)"):
                    show = X_val.copy()
                    show["Actual"] = y_val.values
                    show["Predicted"] = r.y_pred
                    if r.y_proba is not None:
                        show["Score"] = r.y_proba
                    st.dataframe(show.head(50), use_container_width=True)


def main() -> None:
    WineApp().run()


if __name__ == "__main__":
    main()
