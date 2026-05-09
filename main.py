import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)
from wordcloud import WordCloud

from clean import clean_df


class SpamHamModel:
    def __init__(self):
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.df_raw = None
        self.df_clean = None
        self.df_train = None
        self.df_test = None
        self.wi_spam = defaultdict(int)
        self.wi_ham = defaultdict(int)
        self.P_S_tr = 0.0
        self.P_H_tr = 0.0
        self.n_spam_tr = 0
        self.n_ham_tr = 0
        self.y_true = None
        self.y_probs = None
        self.loaded = False

    def load_and_train(self):
        df = pd.read_csv("Data/SpamHam.csv", encoding="latin-1", sep=";")
        df["Message_Length"] = df["SMS_TEXT"].fillna("").astype(str).apply(len)
        self.df_raw = df.copy()

        df_c = df.copy()
        df_c = clean_df(df_c, self.stemmer, self.stop_words)
        df_c["Message_Length"] = df_c["SMS_TEXT"].fillna("").astype(str).apply(len)
        self.df_clean = df_c

        self.df_train, self.df_test = train_test_split(
            df_c, test_size=0.2, random_state=42, stratify=df_c["Label"]
        )

        self.n_spam_tr = int((self.df_train["Label"] == "spam").sum())
        self.n_ham_tr = int((self.df_train["Label"] == "ham").sum())
        n_total = len(self.df_train)
        self.P_S_tr = self.n_spam_tr / n_total
        self.P_H_tr = self.n_ham_tr / n_total

        self.wi_spam = defaultdict(int)
        self.wi_ham = defaultdict(int)
        for msg in self.df_train[self.df_train["Label"] == "spam"]["SMS_TEXT"]:
            for word in set(str(msg).split()):
                self.wi_spam[word] += 1
        for msg in self.df_train[self.df_train["Label"] == "ham"]["SMS_TEXT"]:
            for word in set(str(msg).split()):
                self.wi_ham[word] += 1

        self.y_true = (self.df_test["Label"] == "spam").astype(int).values
        self.y_probs = np.array([
            self._prob_tokens(self.preprocess(str(t)))
            for t in self.df_test["SMS_TEXT"]
        ])
        self.loaded = True

    def preprocess(self, text):
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens if w.isalnum()]
        tokens = [w for w in tokens if w not in self.stop_words]
        return [self.stemmer.stem(w) for w in tokens]

    def _p_ws(self, w):
        return (self.wi_spam[w] + 1) / (self.n_spam_tr + 2)

    def _p_wh(self, w):
        return (self.wi_ham[w] + 1) / (self.n_ham_tr + 2)

    def _prob_tokens(self, tokens):
        if not tokens:
            return 0.5
        probs = [
            (self._p_ws(w) * self.P_S_tr) /
            (self._p_ws(w) * self.P_S_tr + self._p_wh(w) * self.P_H_tr)
            for w in tokens
        ]
        log_num = sum(np.log(p + 1e-300) for p in probs)
        log_den = sum(np.log(1 - p + 1e-300) for p in probs)
        return float(1.0 / (1.0 + np.exp(log_den - log_num)))

    def classify_detailed(self, text):
        tokens = self.preprocess(str(text))
        rows = []
        for w in tokens:
            p_ws = self._p_ws(w)
            p_wh = self._p_wh(w)
            denom = p_ws * self.P_S_tr + p_wh * self.P_H_tr
            p_sw = (p_ws * self.P_S_tr) / denom if denom > 0 else 0.5
            rows.append({
                "word": w,
                "spam_count": self.wi_spam[w],
                "ham_count": self.wi_ham[w],
                "p_ws": p_ws,
                "p_wh": p_wh,
                "p_sw": p_sw,
            })
        return tokens, rows, self._prob_tokens(tokens)

    def get_metrics(self, threshold=0.5):
        y_pred = (self.y_probs >= threshold).astype(int)
        cm = confusion_matrix(self.y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            "precision": precision_score(self.y_true, y_pred, zero_division=0),
            "recall":    recall_score(self.y_true, y_pred, zero_division=0),
            "f1":        f1_score(self.y_true, y_pred, zero_division=0),
            "accuracy":  (tp + tn) / (tp + tn + fp + fn),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }

    # ── Figure builders (return Figure, caller saves/opens) ──────────────────

    def fig_distribution(self, df, title_suffix=""):
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df["Label"].value_counts()
        colors = ["tomato" if l == "spam" else "steelblue" for l in counts.index]
        bars = ax.bar(counts.index, counts.values, color=colors)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                    str(val), ha="center", va="bottom", fontweight="bold")
        ax.set_title(f"Spam vs Ham Distribution {title_suffix}")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        plt.tight_layout()
        return fig

    def fig_avg_length(self, df, title_suffix=""):
        avg_spam = df[df["Label"] == "spam"]["Message_Length"].mean()
        avg_ham  = df[df["Label"] == "ham"]["Message_Length"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(["Spam", "Ham"], [avg_spam, avg_ham], color=["tomato", "seagreen"])
        for bar, val in zip(bars, [avg_spam, avg_ham]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontweight="bold")
        ax.set_title(f"Average Message Length {title_suffix}")
        ax.set_xlabel("Characters")
        plt.tight_layout()
        return fig

    def fig_top_words(self, df, title_suffix=""):
        def top(msgs, n=20):
            return Counter(" ".join(msgs).split()).most_common(n)

        t_spam = top(df[df["Label"] == "spam"]["SMS_TEXT"].fillna("").astype(str))
        t_ham  = top(df[df["Label"] == "ham"]["SMS_TEXT"].fillna("").astype(str))

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].barh([w for w, _ in t_spam[::-1]], [c for _, c in t_spam[::-1]], color="tomato")
        axes[0].set_title(f"Top 20 Spam Words {title_suffix}")
        axes[0].set_xlabel("Count")
        axes[1].barh([w for w, _ in t_ham[::-1]], [c for _, c in t_ham[::-1]], color="seagreen")
        axes[1].set_title(f"Top 20 Ham Words {title_suffix}")
        axes[1].set_xlabel("Count")
        plt.tight_layout()
        return fig

    def fig_wordclouds(self, df, title_suffix=""):
        spam_text = " ".join(df[df["Label"] == "spam"]["SMS_TEXT"].fillna("").astype(str))
        ham_text  = " ".join(df[df["Label"] == "ham"]["SMS_TEXT"].fillna("").astype(str))
        wc_spam = WordCloud(width=800, height=400, background_color="white",
                            colormap="Reds").generate(spam_text)
        wc_ham  = WordCloud(width=800, height=400, background_color="white",
                            colormap="Greens").generate(ham_text)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        axes[0].imshow(wc_spam, interpolation="bilinear")
        axes[0].set_title(f"Spam Word Cloud {title_suffix}")
        axes[0].axis("off")
        axes[1].imshow(wc_ham, interpolation="bilinear")
        axes[1].set_title(f"Ham Word Cloud {title_suffix}")
        axes[1].axis("off")
        plt.tight_layout()
        return fig

    def fig_confusion_matrix(self, threshold=0.5):
        y_pred = (self.y_probs >= threshold).astype(int)
        cm = confusion_matrix(self.y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["HAM", "SPAM"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
        plt.tight_layout()
        return fig

    def fig_threshold_analysis(self):
        thresholds = np.arange(0.05, 1.0, 0.05)
        precisions, recalls, f1s = [], [], []
        for t in thresholds:
            yp = (self.y_probs >= t).astype(int)
            precisions.append(precision_score(self.y_true, yp, zero_division=0))
            recalls.append(recall_score(self.y_true, yp, zero_division=0))
            f1s.append(f1_score(self.y_true, yp, zero_division=0))
        best = int(np.argmax(f1s))
        best_t = float(thresholds[best])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, precisions, "o-", label="Precision", markersize=5)
        ax.plot(thresholds, recalls,    "s-", label="Recall",    markersize=5)
        ax.plot(thresholds, f1s,        "^-", label="F1-score",  markersize=5, linewidth=2)
        ax.axvline(best_t, color="gray", linestyle="--", label=f"Best ({best_t:.2f})")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Precision, Recall & F1 by Classification Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, best_t, precisions[best], recalls[best], f1s[best]
