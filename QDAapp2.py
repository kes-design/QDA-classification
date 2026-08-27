"""
QDA Classifier - Streamlit web app
=====================================
Upload a "database" Excel file (each sheet = one class, rows = samples,
columns = element concentrations) to train a Quadratic Discriminant
Analysis model, then upload an "unknown samples" Excel file to classify
each row.

Run with:
    streamlit run qda_app.py

Requires: streamlit, pandas, numpy, scikit-learn, plotly, openpyxl
Install with:
    pip install streamlit pandas numpy scikit-learn plotly openpyxl
"""

import io
import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

# Fixed colors for the known glass classes; anything else falls back to the pool below.
FIXED_CLASS_COLORS = {
    "Flatglas": "green",
    "Telefoonglas": "red",
    "Verpakkingsglas": "blue",
}
FALLBACK_COLOR_POOL = ["purple", "orange", "brown", "magenta", "gray", "teal", "gold",
                        "blue", "green", "red"]

st.set_page_config(page_title="QDA Classifier", layout="wide")
st.title("QDA Classifier")

st.markdown(
    """
Upload a **database file** where each sheet is a class (e.g. `Flatglas`,
`Verpakkingsglas`, `Telefoonglas`) and rows are samples with element
concentrations as columns. Then upload an **unknown samples file** with
the same element columns to classify.
"""
)

# ---------------------------------------------------------------
# Session state so results survive re-runs / interactions
# ---------------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.feature_cols = None
    st.session_state.class_labels = None
    st.session_state.log_transform = None
    st.session_state.prob_threshold = None
    st.session_state.random_state = None
    st.session_state.pca = None
    st.session_state.viz_qda = None
    st.session_state.X_pca = None
    st.session_state.X_ref = None
    st.session_state.y_ref = None


def _class_color_map(classes):
    pool = itertools.cycle(FALLBACK_COLOR_POOL)
    color_map = {}
    for c in classes:
        if c in FIXED_CLASS_COLORS:
            color_map[c] = FIXED_CLASS_COLORS[c]
        else:
            color_map[c] = next(pool)
    return color_map


def plot_class_regions(pca, viz_qda, X_pca, y, unknown_pca=None, unknown_labels=None,
                        unknown_ids=None, highlight_idx=None, grid_res=180):
    classes = sorted(y.unique())
    n = len(classes)
    color_map = _class_color_map(classes)

    pad_x = 0.08 * ((X_pca[:, 0].max() - X_pca[:, 0].min()) or 1)
    pad_y = 0.08 * ((X_pca[:, 1].max() - X_pca[:, 1].min()) or 1)
    x_min, x_max = X_pca[:, 0].min() - pad_x, X_pca[:, 0].max() + pad_x
    y_min, y_max = X_pca[:, 1].min() - pad_y, X_pca[:, 1].max() + pad_y
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res))
    class_index = {c: i for i, c in enumerate(classes)}
    Z_labels = viz_qda.predict(np.c_[xx.ravel(), yy.ravel()])
    Zi = np.array([class_index[c] for c in Z_labels]).reshape(xx.shape)

    colorscale = []
    for i, c in enumerate(classes):
        colorscale.append([i / n, color_map[c]])
        colorscale.append([(i + 1) / n, color_map[c]])

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xx[0], y=yy[:, 0], z=Zi,
        colorscale=colorscale, zmin=0, zmax=n - 1,
        showscale=False, opacity=0.22, hoverinfo="skip",
    ))
    for c in classes:
        mask = (y == c).values
        fig.add_trace(go.Scatter(
            x=X_pca[mask, 0], y=X_pca[mask, 1], mode="markers",
            marker=dict(color=color_map[c], size=8, line=dict(color="black", width=0.5)),
            name=f"{c} (n={mask.sum()})",
            hovertemplate=f"Class: {c}<br>PC1: %{{x:.2f}} | PC2: %{{y:.2f}}<extra></extra>",
        ))
    if unknown_pca is not None and len(unknown_pca):
        labels = list(unknown_labels) if unknown_labels is not None else [""] * len(unknown_pca)
        ids = list(unknown_ids) if unknown_ids is not None else [str(i + 1) for i in range(len(unknown_pca))]
        customdata = list(zip(ids, labels))
        fig.add_trace(go.Scatter(
            x=unknown_pca[:, 0], y=unknown_pca[:, 1], mode="markers",
            marker=dict(symbol="circle", size=10, color="white", line=dict(width=2, color="black")),
            customdata=customdata, name="Unknown sample(s)",
            hovertemplate="Sample: %{customdata[0]}<br>Predicted: %{customdata[1]}"
                           "<br>PC1: %{x:.2f} | PC2: %{y:.2f}<extra></extra>",
        ))
        if highlight_idx is not None and 0 <= highlight_idx < len(unknown_pca):
            fig.add_trace(go.Scatter(
                x=[unknown_pca[highlight_idx, 0]], y=[unknown_pca[highlight_idx, 1]],
                mode="markers",
                marker=dict(symbol="circle-open", size=26, color="gold", line=dict(width=4, color="gold")),
                name="Selected sample", hoverinfo="skip", showlegend=False,
            ))
    fig.update_layout(
        title="How the classes are divided (2D PCA projection)",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} of variance)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=550, margin=dict(t=80),
    )
    return fig


# =================================================================
# STEP 1 — Upload database & train
# =================================================================
st.header("1. Train on your database")

db_file = st.file_uploader("Database Excel file (.xlsx)", type=["xlsx"], key="db")

if db_file is not None:
    try:
        xls = pd.ExcelFile(db_file)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        st.stop()

    sheet_names = xls.sheet_names
    st.write(f"Found {len(sheet_names)} sheets (= classes): {', '.join(sheet_names)}")

    selected_sheets = st.multiselect("Sheets to use as classes", sheet_names, default=sheet_names)

    if len(selected_sheets) < 2:
        st.warning("Select at least 2 sheets/classes to run QDA.")
    else:
        preview_cols = list(pd.read_excel(xls, sheet_name=selected_sheets[0], nrows=0).columns)
        id_col_choice = st.selectbox(
            "Sample ID column to exclude from training (not a measurement)",
            options=["(none)"] + preview_cols,
            index=1 if preview_cols else 0,
            help="Pick the column that identifies each sample (e.g. Sample_ID), "
                 "so it isn't treated as an element measurement. Defaults to the first column.",
        )

        log_transform = st.checkbox(
            "Log10-transform concentrations", value=True,
            help="Recommended: trace-element concentrations are typically right-skewed; "
                 "QDA assumes each class is multivariate normal.",
        )
        reg_param = st.slider(
            "Regularization (reg_param)", 0.0, 1.0, 0.0, 0.01,
            help="0 = full QDA (separate covariance per class). Increase toward 1 if a "
                 "class has few samples relative to the number of elements.",
        )
        cv_folds = st.slider("Cross-validation folds", 2, 10, 10, 1)
        prob_threshold = st.slider(
            "Confidence threshold", 0.5, 1.0, 0.90, 0.01,
            help="Posterior probability below which a prediction is flagged as not confident.",
        )
        random_state = st.number_input(
            "Random state", min_value=0, max_value=10_000, value=42, step=1,
            help="Seed used for the cross-validation shuffling and the PCA visualization. "
                 "Change it to check how sensitive the results are, or keep it fixed for reproducibility.",
        )

        if st.button("Train model", type="primary"):
            frames = []
            for sheet in selected_sheets:
                sheet_df = pd.read_excel(xls, sheet_name=sheet)
                sheet_df = sheet_df.dropna(how="all")
                sheet_df["label"] = sheet
                frames.append(sheet_df)
            data = pd.concat(frames, ignore_index=True)

            excluded_cols = {"label"}
            if id_col_choice != "(none)":
                excluded_cols.add(id_col_choice)
            feature_cols = [c for c in data.columns if c not in excluded_cols]

            X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
            y = data["label"]

            mask = X.notna().all(axis=1) & y.notna()
            n_dropped = (~mask).sum()
            if n_dropped:
                st.warning(f"Dropped {n_dropped} row(s) with missing/non-numeric values.")
            X, y = X[mask], y[mask]

            dropped_log = []
            if log_transform:
                X = np.log10(X.replace(0, np.nan))
                keep = X.dropna().index
                dropped_log = sorted(set(X.index) - set(keep))
                X, y = X.loc[keep], y.loc[keep]
                if dropped_log:
                    st.warning(
                        f"Dropped {len(dropped_log)} more row(s) with zero/invalid values "
                        f"before the log10 transform."
                    )

            st.write(f"Using {len(feature_cols)} feature columns: {', '.join(feature_cols)}")
            st.write("Class counts:")
            st.dataframe(y.value_counts().rename("samples"))

            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X)

            min_class_size = y.value_counts().min()
            folds = min(cv_folds, min_class_size)
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
            qda_cv = QuadraticDiscriminantAnalysis(reg_param=reg_param)
            y_pred_cv = cross_val_predict(qda_cv, Xs, y, cv=skf)

            labels = sorted(y.unique())

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cross-validated accuracy", f"{accuracy_score(y, y_pred_cv):.1%}")
                st.caption(f"Stratified {folds}-fold cross-validation.")
                st.text("Classification report:")
                st.text(classification_report(y, y_pred_cv, target_names=labels))
            with col2:
                cm = confusion_matrix(y, y_pred_cv, labels=labels)
                st.text("Confusion matrix (rows = true, columns = predicted):")
                st.dataframe(pd.DataFrame(cm, index=labels, columns=labels))

            # Final model trained on ALL data (used for real classification)
            final_qda = QuadraticDiscriminantAnalysis(reg_param=reg_param)
            final_qda.fit(Xs, y)

            st.session_state.model = final_qda
            st.session_state.scaler = scaler
            st.session_state.feature_cols = feature_cols
            st.session_state.class_labels = labels
            st.session_state.log_transform = log_transform
            st.session_state.prob_threshold = prob_threshold
            st.session_state.random_state = random_state

            st.success(
                "Final model trained on all available data. "
                "You can now classify unknown samples below."
            )

            # ---------------------------------------------------
            # Example class-region plot, analogous to the example
            # tree visualization in the Random Forest app
            # ---------------------------------------------------
            st.subheader("How the classes are divided (example visualization)")
            st.caption(
                "The real model uses all features at once, which can't be drawn directly. "
                "This plot compresses the data to its 2 strongest principal components "
                "and shows the region each class would occupy — an approximation of the "
                "true decision boundary, useful for building intuition."
            )
            pca = PCA(n_components=2, random_state=random_state).fit(Xs)
            X_pca = pca.transform(Xs)
            viz_qda = QuadraticDiscriminantAnalysis(reg_param=reg_param).fit(X_pca, y)

            st.session_state.pca = pca
            st.session_state.viz_qda = viz_qda
            st.session_state.X_pca = X_pca
            st.session_state.X_ref = X
            st.session_state.y_ref = y

            st.plotly_chart(plot_class_regions(pca, viz_qda, X_pca, y), use_container_width=True)


# =================================================================
# STEP 2 — Upload unknown samples & classify
# =================================================================
st.header("2. Classify unknown samples")

if st.session_state.model is None:
    st.info("Train a model in step 1 first.")
else:
    tab_upload, tab_manual = st.tabs(["Upload a file", "Enter one sample manually"])
    feature_cols = st.session_state.feature_cols
    scaler = st.session_state.scaler
    model = st.session_state.model
    log_transform = st.session_state.log_transform
    prob_threshold = st.session_state.prob_threshold

    def _classify(unknown_df):
        missing_cols = [c for c in feature_cols if c not in unknown_df.columns]
        if missing_cols:
            st.error(f"Unknown samples file is missing required columns: {missing_cols}")
            return None

        X_unk = unknown_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        if log_transform:
            X_unk = np.log10(X_unk.replace(0, np.nan))

        valid_mask = X_unk.notna().all(axis=1)
        n_invalid = (~valid_mask).sum()
        if n_invalid:
            st.warning(f"Skipping {n_invalid} row(s) with missing/invalid feature values.")

        X_unk_valid = X_unk[valid_mask]
        if len(X_unk_valid) == 0:
            st.error("No valid rows to classify after cleaning.")
            return None

        Xs_unk = scaler.transform(X_unk_valid)
        pred = model.predict(Xs_unk)
        proba = model.predict_proba(Xs_unk)
        proba_df = pd.DataFrame(proba, index=X_unk_valid.index, columns=model.classes_)

        results = unknown_df.loc[valid_mask].copy()
        results["Predicted_class"] = pred
        results["posterior_probability"] = proba_df.max(axis=1).values
        results["confident"] = results["posterior_probability"] >= prob_threshold
        for cls in model.classes_:
            results[f"P({cls})"] = proba_df[cls].values
        return results

    with tab_upload:
        unknown_file = st.file_uploader(
            "Unknown samples Excel/CSV file (.xlsx or .csv)", type=["xlsx", "csv"], key="unknown"
        )
        if unknown_file is not None:
            if unknown_file.name.lower().endswith(".csv"):
                unknown_df = pd.read_csv(unknown_file)
            else:
                unknown_df = pd.read_excel(unknown_file)
            unknown_df = unknown_df.dropna(how="all")

            results = _classify(unknown_df)
            if results is not None:
                def _highlight(row):
                    return ["background-color: #fff3cd" if not row["confident"] else "" for _ in row]

                st.subheader("Results")
                st.caption("Click a row to highlight that sample in the projection below.")
                selection_event = st.dataframe(
                    results.style.apply(_highlight, axis=1).format(
                        {"posterior_probability": "{:.4f}"}
                        | {c: "{:.4f}" for c in results.columns if c.startswith("P(")}
                    ),
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="results_table",
                )
                selected_rows = (
                    selection_event.selection.rows
                    if selection_event is not None and selection_event.selection is not None
                    else []
                )
                highlight_idx = selected_rows[0] if selected_rows else None

                n_low = (~results["confident"]).sum()
                if n_low:
                    st.warning(f"{n_low} sample(s) fell below the {prob_threshold:.0%} confidence threshold.")

                # Build the PCA points from the SAME rows/order as `results` (i.e. only the
                # valid, classified rows) so the table's row selection lines up correctly
                # with the corresponding point below.
                X_unk_for_pca = unknown_df.loc[results.index, feature_cols].apply(pd.to_numeric, errors="coerce")
                if log_transform:
                    X_unk_for_pca = np.log10(X_unk_for_pca.replace(0, np.nan))
                unk_pca = st.session_state.pca.transform(scaler.transform(X_unk_for_pca))
                # Use a non-feature column (e.g. Sample_ID) to label unknown points on hover,
                # falling back to a simple row number if no such column exists.
                id_cols = [c for c in unknown_df.columns if c not in feature_cols]
                sample_ids = results[id_cols[0]] if id_cols else (results.index + 1)

                st.plotly_chart(
                    plot_class_regions(
                        st.session_state.pca, st.session_state.viz_qda, st.session_state.X_pca,
                        st.session_state.y_ref, unknown_pca=unk_pca,
                        unknown_labels=results["Predicted_class"],
                        unknown_ids=sample_ids,
                        highlight_idx=highlight_idx,
                    ),
                    use_container_width=True,
                )

                buffer = io.BytesIO()
                results.to_csv(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    "Download results as CSV",
                    data=buffer,
                    file_name="classification_results.csv",
                    mime="text/csv",
                )

    with tab_manual:
        st.caption("Enter concentrations for a single sample (same units as the reference database).")
        cols = st.columns(4)
        manual_values = {}
        for i, el in enumerate(feature_cols):
            with cols[i % 4]:
                manual_values[el] = st.number_input(el, value=0.0, format="%.6f", key=f"manual_{el}")

        if st.button("Classify this sample"):
            if all(v == 0.0 for v in manual_values.values()):
                st.warning("Enter at least some non-zero concentrations first.")
            else:
                manual_df = pd.DataFrame([manual_values])
                results = _classify(manual_df)
                if results is not None:
                    st.dataframe(results, use_container_width=True)
                    pred = results.loc[0, "Predicted_class"]
                    prob = results.loc[0, "posterior_probability"]
                    st.success(f"Predicted class: **{pred}** (posterior probability: {prob:.1%})")