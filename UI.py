from __future__ import annotations
import threading
import tempfile
import subprocess

from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Input, Button, Label, Static,
    TabbedContent, TabPane, DataTable, LoadingIndicator,
)
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual import on

from main import SpamHamModel

model = SpamHamModel()


def _open_figure(fig) -> None:
    import matplotlib.pyplot as plt
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    subprocess.Popen(["open", path])


class SpamHamApp(App):
    TITLE = "Spam / Ham Bayesian Classifier"
    SUB_TITLE = "Naive Bayes · Laplace Smoothing · EDA"

    CSS = """
    #loader-container {
        align: center middle;
        height: 100%;
    }
    #loader-label {
        margin-top: 1;
        color: $text-muted;
        text-align: center;
    }
    #tabs {
        display: none;
    }
    .tab-scroll {
        padding: 1 2;
        height: 1fr;
    }
    .section-title {
        background: $primary;
        color: $text;
        padding: 0 1;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
    }
    .stats-box {
        border: round $primary;
        padding: 1 2;
        margin-bottom: 1;
    }
    .chart-row {
        height: auto;
        margin-bottom: 1;
    }
    .chart-row Button {
        margin-right: 1;
    }
    #result-box {
        border: round $primary;
        padding: 1 2;
        margin-top: 1;
        min-height: 4;
    }
    #result-box.spam {
        border: round $error;
        color: $error;
    }
    #result-box.ham {
        border: round $success;
        color: $success;
    }
    #word-table {
        height: 16;
        margin-top: 1;
    }
    #word-table-top {
        height: 4;
        margin-top: 1;
    }
    #metrics-table {
        height: 7;
    }
    #cm-box {
        border: round $primary;
        padding: 1 2;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical(id="loader-container"):
            yield LoadingIndicator()
            yield Label(
                "Loading dataset and training Bayesian classifier…",
                id="loader-label",
            )

        with TabbedContent(id="tabs", initial="tab-classify"):

            # ── Classify ──────────────────────────────────────────────────────
            with TabPane("  Classify  ", id="tab-classify"):
                with VerticalScroll(classes="tab-scroll"):
                    yield Label("Text Message Classifier", classes="section-title")
                    yield Input(
                        placeholder="Enter a text message to classify…",
                        id="text-input",
                    )
                    with Horizontal(classes="chart-row"):
                        yield Button("Analyze", id="btn-analyze", variant="primary")
                        yield Button("Clear", id="btn-clear")
                    yield Static(
                        "Enter a message above and press Analyze.",
                        id="result-box",
                    )
                    yield Label("Top 3 Words by Wheight breakdown:", classes="section-title")
                    wt_top = DataTable(id="word-table-top", zebra_stripes=True)
                    wt_top.add_columns(
                        "Word", "# Spam msgs", "# Ham msgs",
                        "P(W|S)", "P(W|H)", "P(S|W)",
                    )
                    yield wt_top
                    
                    yield Label("Per-word Bayesian breakdown:", classes="section-title")
                    wt = DataTable(id="word-table", zebra_stripes=True)
                    wt.add_columns(
                        "Word", "# Spam msgs", "# Ham msgs",
                        "P(W|S)", "P(W|H)", "P(S|W)",
                    )
                    yield wt

                    # yield Label("Note: Probabilities are estimated with Laplace smoothing.", classes="stats-box")
                    

            # ── Raw EDA ───────────────────────────────────────────────────────
            with TabPane("  Raw EDA  ", id="tab-raw"):
                with VerticalScroll(classes="tab-scroll"):
                    yield Label(
                        "Exploratory Data Analysis — Raw Data",
                        classes="section-title",
                    )
                    yield Static("", id="stats-raw", classes="stats-box")
                    yield Label("Charts", classes="section-title")
                    with Horizontal(classes="chart-row"):
                        yield Button("Distribution",  id="btn-dist-raw")
                        yield Button("Avg Length",    id="btn-len-raw")
                        yield Button("Top 20 Words",  id="btn-words-raw")
                        yield Button("Word Clouds",   id="btn-clouds-raw")

            # ── Cleaned EDA ───────────────────────────────────────────────────
            with TabPane("  Cleaned EDA  ", id="tab-clean"):
                with VerticalScroll(classes="tab-scroll"):
                    yield Label(
                        "Exploratory Data Analysis — Cleaned Data",
                        classes="section-title",
                    )
                    yield Static("", id="stats-clean", classes="stats-box")
                    yield Label("Charts", classes="section-title")
                    with Horizontal(classes="chart-row"):
                        yield Button("Distribution",  id="btn-dist-clean")
                        yield Button("Avg Length",    id="btn-len-clean")
                        yield Button("Top 20 Words",  id="btn-words-clean")
                        yield Button("Word Clouds",   id="btn-clouds-clean")

            # ── Model Evaluation ──────────────────────────────────────────────
            with TabPane("  Model Eval  ", id="tab-eval"):
                with VerticalScroll(classes="tab-scroll"):
                    yield Label("Model Evaluation", classes="section-title")
                    yield Static("", id="stats-eval", classes="stats-box")
                    yield Label(
                        "Metrics at threshold = 0.5", classes="section-title"
                    )
                    mt = DataTable(id="metrics-table", zebra_stripes=True)
                    mt.add_columns("Metric", "Score")
                    yield mt
                    yield Label("Confusion Matrix", classes="section-title")
                    yield Static("", id="cm-box")
                    yield Label("Charts", classes="section-title")
                    with Horizontal(classes="chart-row"):
                        yield Button("Confusion Matrix",   id="btn-cm")
                        yield Button("Threshold Analysis", id="btn-thresh")

        yield Footer()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.run_worker(self._load_model, thread=True, name="model_loader")

    def _load_model(self) -> None:
        model.load_and_train()
        self.call_from_thread(self._on_loaded)

    def _on_loaded(self) -> None:
        self.query_one("#loader-container").display = False
        self.query_one("#tabs").display = True
        self._fill_eda_stats()
        self._fill_eval_stats()

    # ── Stats population ──────────────────────────────────────────────────────

    def _fill_eda_stats(self) -> None:
        for kind, df in (("raw", model.df_raw), ("clean", model.df_clean)):
            n  = len(df)
            ns = (df["Label"] == "spam").sum()
            nh = (df["Label"] == "ham").sum()
            as_ = df[df["Label"] == "spam"]["Message_Length"].mean()
            ah  = df[df["Label"] == "ham"]["Message_Length"].mean()
            self.query_one(f"#stats-{kind}", Static).update(
                f"Total: {n}  |  Spam: {ns} ({ns/n*100:.1f}%)  |  Ham: {nh} ({nh/n*100:.1f}%)\n"
                f"Avg spam length: {as_:.1f} chars    Avg ham length: {ah:.1f} chars"
            )

    def _fill_eval_stats(self) -> None:
        n_te = len(model.df_test)
        ns_te = (model.df_test["Label"] == "spam").sum()
        nh_te = (model.df_test["Label"] == "ham").sum()
        self.query_one("#stats-eval", Static).update(
            f"Train : {len(model.df_train)} messages "
            f"(spam={model.n_spam_tr}, ham={model.n_ham_tr})\n"
            f"Test  : {n_te} messages (spam={ns_te}, ham={nh_te})\n"
            f"Prior  P(Spam) = {model.P_S_tr:.4f}   P(Ham) = {model.P_H_tr:.4f}"
        )
        m  = model.get_metrics(0.5)
        mt = self.query_one("#metrics-table", DataTable)
        mt.add_row("Precision (SPAM)", f"{m['precision']:.4f}")
        mt.add_row("Recall    (SPAM)", f"{m['recall']:.4f}")
        mt.add_row("F1-score  (SPAM)", f"{m['f1']:.4f}")
        mt.add_row("Accuracy",         f"{m['accuracy']:.4f}")
        self.query_one("#cm-box", Static).update(
            f"  TP (spam → spam correctly) : {m['tp']:4d}   "
            f"FN (spam → ham, missed)  : {m['fn']:4d}\n"
            f"  FP (ham  → spam, false alarm): {m['fp']:4d}   "
            f"TN (ham  → ham correctly) : {m['tn']:4d}"
        )

    # ── Button handlers ───────────────────────────────────────────────────────

    @on(Button.Pressed, "#btn-analyze")
    def _analyze(self) -> None:
        if not model.loaded:
            return
        text = self.query_one("#text-input", Input).value.strip()
        if not text:
            self.query_one("#result-box", Static).update("Please enter a message.")
            return
        tokens, rows, prob = model.classify_detailed(text)
        label = "SPAM" if prob >= 0.5 else "HAM"
        icon  = "⚠" if label == "SPAM" else "✓"
        rb = self.query_one("#result-box", Static)
        rb.update(
            f"{icon}  {label}   (P(SPAM) = {prob:.6f})\n"
            f"Tokens: {', '.join(tokens) if tokens else '(none after preprocessing)'}"
        )
        rb.remove_class("spam")
        rb.remove_class("ham")
        rb.add_class(label.lower())
        wt_top = self.query_one("#word-table-top", DataTable)
        wt_top.clear()
        # show top 3 words by weight (P(S|W) if available)
        top_rows = sorted(rows, key=lambda r: r.get("p_sw", 0), reverse=True)[:3]
        for r in top_rows:
            wt_top.add_row(
                r["word"],
                str(r["spam_count"]),
                str(r["ham_count"]),
                f"{r['p_ws']:.5f}",
                f"{r['p_wh']:.5f}",
                f"{r['p_sw']:.5f}",
            )

        wt = self.query_one("#word-table", DataTable)
        wt.clear()
        for r in rows:
            wt.add_row(
                r["word"],
                str(r["spam_count"]),
                str(r["ham_count"]),
                f"{r['p_ws']:.5f}",
                f"{r['p_wh']:.5f}",
                f"{r['p_sw']:.5f}",
            )

    @on(Button.Pressed, "#btn-clear")
    def _clear(self) -> None:
        self.query_one("#text-input", Input).value = ""
        rb = self.query_one("#result-box", Static)
        rb.update("Enter a message above and press Analyze.")
        rb.remove_class("spam")
        rb.remove_class("ham")
        self.query_one("#word-table", DataTable).clear()
        self.query_one("#word-table-top", DataTable).clear()

    # ── Chart helpers ─────────────────────────────────────────────────────────

    def _bg(self, fn, *args) -> None:
        threading.Thread(
            target=lambda: _open_figure(fn(*args)), daemon=True
        ).start()

    @on(Button.Pressed, "#btn-dist-raw")
    def _dist_raw(self):    self._bg(model.fig_distribution, model.df_raw,   "(Raw)")

    @on(Button.Pressed, "#btn-dist-clean")
    def _dist_clean(self):  self._bg(model.fig_distribution, model.df_clean, "(Cleaned)")

    @on(Button.Pressed, "#btn-len-raw")
    def _len_raw(self):     self._bg(model.fig_avg_length,   model.df_raw,   "(Raw)")

    @on(Button.Pressed, "#btn-len-clean")
    def _len_clean(self):   self._bg(model.fig_avg_length,   model.df_clean, "(Cleaned)")

    @on(Button.Pressed, "#btn-words-raw")
    def _words_raw(self):   self._bg(model.fig_top_words,    model.df_raw,   "(Raw)")

    @on(Button.Pressed, "#btn-words-clean")
    def _words_clean(self): self._bg(model.fig_top_words,    model.df_clean, "(Cleaned)")

    @on(Button.Pressed, "#btn-clouds-raw")
    def _clouds_raw(self):  self._bg(model.fig_wordclouds,   model.df_raw,   "(Raw)")

    @on(Button.Pressed, "#btn-clouds-clean")
    def _clouds_clean(self): self._bg(model.fig_wordclouds,  model.df_clean, "(Cleaned)")

    @on(Button.Pressed, "#btn-cm")
    def _cm(self):          self._bg(model.fig_confusion_matrix)

    @on(Button.Pressed, "#btn-thresh")
    def _thresh(self):
        def run():
            fig, *_ = model.fig_threshold_analysis()
            _open_figure(fig)
        threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    SpamHamApp().run()
