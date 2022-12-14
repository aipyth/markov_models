import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import json
import subprocess


def read_text(filename, keep_spaces=True):
    text = ""
    with open(filename) as f:
        text += "".join(f.readlines())
    text = text.lower()
    text = re.sub(
        r"[^а-яґєіїйщюя \n]" if keep_spaces else r"[^а-яґєіїйщюя]",
        r"",
        text,
    )
    text = re.sub(r"[ÑÑ]", r"", text)
    text = re.sub(r"\n", r" ", text)
    text = re.sub(r"  ", r" ", text)
    return text


def get_frequencies(data):
    frequencies = {}
    for letter in data:
        if letter not in frequencies:
            frequencies[letter] = 0
        frequencies[letter] += 1
    for k in frequencies:
        frequencies[k] /= len(data)
    return frequencies


def scatter_frequencies(frequencies: dict):
    fq = np.array(list(frequencies.items()))
    x = np.arange(fq.shape[0])
    y = np.array(fq[:, 1], dtype=float)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, d in enumerate(fq.tolist()):
        ax.annotate(d[0], xy=(x[i], y[i]), xytext=(x[i] - 0.35, y[i] + y.max() * 0.03))
    return fig, ax


def make_stochastic_matrix(size, generator="normal"):
    if generator == "normal":
        M = np.random.randn(*size)
        M = 0.2 * (M + 2 * abs(M.min()))
    elif generator == "evenly":
        M = np.full(size, 1 / size if len(size) == 1 else 1 / size[1])
        m = np.random.randn(*size)
        m = 0.2 * (M + 2 * abs(M.min()))
        M += m
        # M /= M.sum(axis=1).reshape
    return M / (M.sum() if len(size) == 1 else M.sum(axis=1).reshape((M.shape[0], 1)))


BAUM_WELCH_EXECUTABLE = "bin/baum_welch"
VITERBI_EXECUTABLE = "bin/viterbi"


class BaumWelchRunner:
    def __init__(self, args: dict):
        self.data = args["data"] if "data" in args else ""
        self.V = args["V"] if "V" in args else []
        self.N = args["N"] if "N" in args else 0
        self.eps = args["eps"] if "eps" in args else 1e-8
        self.min_iterations = args["min_iterations"] if "min_iterations" in args else 35
        self.input_file = args["input_file"] if "input_file" in args else "input.json"
        self.output_file = (
            args["output_file"]
            if "output_file" in args
            else "bw-results/hmm-baum-welch.json"
        )
        self.A = args["A"] if "A" in args else []
        self.B = args["B"] if "B" in args else []
        self.P = args["P"] if "P" in args else []
        self.do_not_re_estimate = []

    def __str__(self) -> str:
        return f"<BaumWelch Runner {self.input_file} -> {self.output_file}>"

    def load_data(self, filename, size=None, keep_spaces=False):
        self.data = read_text(filename, keep_spaces)
        if size is not None:
            self.data = self.data[:size]
        self.frequencies = get_frequencies(self.data)
        self.V = list(self.frequencies.keys())
        return self

    def generate_initial(self, generate: dict, P=None, A=None, B=None):
        self.P = P
        if self.P is None:
            self.P = make_stochastic_matrix(
                (1, self.N), generator=generate["P"] or "normal"
            ).flatten()
        self.A = A
        if self.A is None:
            self.A = make_stochastic_matrix(
                (self.N, self.N), generator=generate["A"] or "normal"
            )
        self.B = B
        if self.B is None:
            self.B = make_stochastic_matrix(
                (self.N, len(self.V)), generator=generate["B"] or "normal"
            )
        return self

    def prepare_input(self):
        c = {
            "data": self.data,
            "V": self.V,
            "N": self.N,
            "eps": self.eps,
            "min_iterations": self.min_iterations,
            "output_file": self.output_file,
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "P": self.P.tolist(),
            "do_not_re_estimate": self.do_not_re_estimate or [],
        }
        with open(self.input_file, "w+") as f:
            json.dump(c, f)
        return self

    def run(self):
        subprocess.run([BAUM_WELCH_EXECUTABLE, self.input_file]).check_returncode()

    def read_results(self):
        res = None
        with open(self.output_file) as f:
            res = json.load(f)
        self.results = {
            "observations": np.array(res["observations"]),
            "encoding": res["encoding"],
            "model": {
                "N": res["model"]["N"],
                "M": res["model"]["M"],
                "A": np.array(res["model"]["A"]),
                "B": pd.DataFrame(np.array(res["model"]["B"]).T, index=self.V),
                "P": np.array(res["model"]["P"]),
            },
            "evolution": {
                "A": np.array(res["evolution"]["a"]),
                "B": np.array(res["evolution"]["b"]),
                "P": np.array(res["evolution"]["p"]),
                "P_LOG": np.array(res["evolution"]["p_log"]),
                "delta": np.array(res["evolution"]["delta"]),
            },
        }
        return self.results

    def iterations(self):
        return len(self.results["evolution"]["B"])

    def show_delta(self):
        delta_hist = self.results["evolution"]["delta"]
        x = np.arange(0, delta_hist.shape[0])
        y = delta_hist
        fig, ax = plt.subplots()
        ax.loglog(x[2:], y[2:])
        ax.grid(linestyle="--", color="gray")
        ax.set_title("Delta")
        fig.show()

    def show_clusters_kmeans(self):
        df = self.results["model"]["B"]
        preds = KMeans(n_clusters=self.N).fit_predict(df)
        clusters = []
        for i in range(self.N):
            clusters.append(np.array(self.V)[np.where(preds == i)[0]])
        return clusters, preds

    def show_clusters_max(self):
        df = self.results["model"]["B"]
        preds = np.array(df.idxmax(axis=1))
        clusters = []
        for i in range(self.N):
            clusters.append(np.array(self.V)[np.where(preds == i)[0]])
        return clusters, preds


class ViterbiRunner:
    def __init__(self, args: dict):
        self.data = args["data"] if "data" in args else ""
        self.N = args["N"] if "N" in args else 0
        self.input_file = args["input_file"] if "input_file" in args else "input.json"
        self.output_file = (
            args["output_file"]
            if "output_file" in args
            else "bw-results/hmm-viterbi.json"
        )
        self.A = args["A"] if "A" in args else []
        self.B = args["B"] if "B" in args else []
        self.P = args["P"] if "P" in args else []

    def prepare_input(self):
        c = {
            "data": self.data,
            "N": self.N,
            "A": self.A,
            "B": self.B,
            "P": self.P,
            "output_file": self.output_file,
        }
        with open(self.input_file, "w+") as f:
            json.dump(c, f)
        return self

    def run(self):
        subprocess.run([VITERBI_EXECUTABLE, self.input_file])

    def read_results(self):
        res = None
        with open(self.output_file) as f:
            res = json.load(f)
        self.results = {
            "delta": res["delta"],
            "q": res["q"],
            "P": res["P"],
        }
        return self.results
