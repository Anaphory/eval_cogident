import re
import json
import itertools
import collections
from io import StringIO

from shutil import copy
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures

import numpy as np

import pandas
import graphviz
from sklearn.linear_model import LogisticRegression, LinearRegression, Lars, LassoLars
from sklearn.svm import SVR
from sklearn.tree.export import export_graphviz, _DOTTreeExporter, _tree, _criterion
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from mlinsights.mlmodel.piecewise_estimator import PiecewiseRegressor
from mlinsights.mlmodel.piecewise_tree_regression import PiecewiseTreeRegressor

from dataset_properties import stats

stats = {
    Path(dataset).stem: data
    for dataset, data in stats.items()
}

rows = []
with open("calculated_scores.txt") as data:
    for line in data:
        run, bcubedfscore, vmeasure, adjustedrand, adjustedmutualinformation = line.split()
        _, dataset, clustering, method_soundclass, parameters, filename = run.split("/")
        clustering_algo, clustering_threshold = clustering.split("-")
        method, soundclass = method_soundclass.split("-")
        if filename == "wordlist.tsv":
            assert method == "lexstat"
            batch_size, alpha = None, None
            initial_threshold, local_weight, gop, align = re.match("(-?[0-9.]+)-(-?[0-9.]+)-(-?[0-9.]+)-([a-z]+)", parameters).groups()
        elif filename == "wordlist.json":
            assert method == "pmi"
            local_weight, gop, align = None, None, None
            initial_threshold, batch_size, alpha = parameters.split("-")
        rows.append((run, dataset,
            stats[dataset]["min_intersection"],
            stats[dataset]["average_intersection"],
            stats[dataset]["average_asjp_edit_distance"],
            stats[dataset]["mean_word_length"],
            stats[dataset]["mean_synonyms"],
            stats[dataset]["quantile_for_two_segments"],
            # In LexiRumah, a morpheme has 5.25 segments in the mean, and 90% of all marked morphemes have 9 segments or fewer.
            stats[dataset]["quantile_for_five_segments"],
            stats[dataset]["quantile_for_nine_segments"],
            # stats[dataset]["n_lects"],
            stats[dataset]["mean_segments"],
            stats[dataset]["mean_asjp_segments"],
            stats[dataset]["mean_asjp_segments_product"],
                     method, soundclass, clustering_algo,
                     int(clustering_threshold)/100., int(initial_threshold)/100.,
                     int(batch_size) if batch_size else None, int(alpha)/100 if
                     alpha else None, float(local_weight) if local_weight else
                     None, float(gop) if gop else None, align,
                     float(bcubedfscore), float(vmeasure), float(adjustedrand),
                     float(adjustedmutualinformation)))

def onehot(df, column, break_at=20):
    x = sorted(set(df[column]) - {None})
    if len(x) > break_at:
        raise ValueError
    unrepresented = x.pop(0)
    c = []
    for v in x:
        c.append("{:s}:{:}".format(column, v))
        df[c[-1]] = (df[column] == v) * 1
    return c


measures = ["B3", "V", "aR", "AMI"]
df = pandas.DataFrame(rows, columns=[
    "Run", "Dataset",
    "min_intersection", "average_intersection", "average_asjp_edit_distance", "mean_word_length", "mean_synonyms", "quantile_for_two_segments", "quantile_for_five_segments", "quantile_for_nine_segments",
    #"n_lects",
    "mean_segments", "mean_asjp_segments", "mean_asjp_segments_product",
    "Method", "Soundclass", "Clustering", "CThreshold", "IThreshold", "Batchsize", "Alpha", "LWeight", "GOP", "Alignment"] + measures)
df.set_index("Run", inplace=True)

for measure in measures:
    df.sort_values(measure, inplace=True)
    df[measure + "_rank"] = range(len(df))

families = {"austronesian": ["D_training_Austronesian-210-20", "data-an-45-210"],
            "indoeuropean": ['D_training_IndoEuropean-207-20', "data-ie-42-208"],
            "romance": ['D_test_Romance-110-43']}
families = {}

for family, datasets in families.items():
    print(family)
    for measure1, measure2 in itertools.combinations(measures, 2):
        print(measure1, measure2)
        max_score = 0
        for ds in datasets:
            print(ds)
            data = df[df.Dataset == ds]
            print(len(data))
            c1 = data[measure1 + "_rank"].values
            c2 = data[measure2 + "_rank"].values
            index = data.index.values

            for i, j in itertools.combinations(range(len(index)), 2):
                score = - (c1[i] - c1[j]) * (c2[i] - c2[j])
                if score > max_score:
                    if c1[i] > c1[j]:
                        d = index[j], index[i]
                    else:
                        d = index[i], index[j]
                    max_score = score
        print()
        path = Path("{:}/{:}-vs-{:}".format(family, measure1, measure2))
        path.mkdir(parents=True, exist_ok=True)
        print(measure1, measure2)
        i, j = d
        with (path / "summary.txt").open("w") as op:
            print(df.loc[i], file=op)
            print(df.loc[j], file=op)
        copy(i, str(path / (measure1 + Path(i).suffix)))
        copy(j, str(path / (measure2 + Path(j).suffix)))


# for ds, row in df.groupby("Dataset").mean().iterrows():
#     means = row[measures]
#     df.loc[df["Dataset"] == ds, measures] -= means
# for ds, row in df.groupby("Dataset").std().iterrows():
#     std = row[measures]
#     df.loc[df["Dataset"] == ds, measures] /= std

c = []
for column in ["Soundclass", "Clustering"]:
    c.extend(onehot(df, column))

class GraphvizPiecewiseExporter(_DOTTreeExporter):
    def __init__(self, *args, **kwargs):
        self.trafo = lambda x: x
        super().__init__(*args, **kwargs)

    def export(self, tree):
        self.estimators_ = {i: tree.estimators_[k]
                            for k, i in enumerate(tree.leaves_)}
        super().export(tree)

    def value_text(self, tree, node_id):
        text = super().value_text(tree, node_id)
        try:
            text += "\n" + str(dict(zip(
                self.feature_names,
                self.estimators_[node_id].coef_))).replace(", ", ",\n")
        except KeyError:
            pass
        return text

class TransformedExporter(_DOTTreeExporter):
    def __init__(self, transform, *args, **kwargs):
        self.trafo = transform
        super().__init__(*args, **kwargs)

    def value_text(self, tree, node_id):
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]
        value = self.trafo(value)

        # Write node class distribution / regression value
        if self.proportion and tree.n_classes[0] != 1:
            # For classification this will show the proportion of samples
            value = value / tree.weighted_n_node_samples[node_id]
        if tree.n_classes[0] == 1:
            # Regression
            value_text = np.around(value, self.precision)
        elif self.proportion:
            # Classification
            value_text = np.around(value, self.precision)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            # Classification without floating-point weights
            value_text = value.astype(int)
        else:
            # Classification with floating-point weights
            value_text = np.around(value, self.precision)
        # Strip whitespace
        value_text = str(value_text.astype('S32')).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", self.characters[4])

        return value_text


more = open("more.sh", "w")

try:
    best_model.get("lexstat")
except (NameError, AttributeError):
    best_model = {}
best_model = {}

def transform(y):
    return y
    return np.exp(y)
def untransform(y_raw):
    return y_raw
    return np.log(y_raw)
def squish(d):
    return d

linearmodeltree = True

for method, parameters in [
        ("pmi", ["Batchsize", "Alpha"]),
        ("lexstat", ["LWeight", "GOP"] + onehot(df, "Alignment")),
]:
    data = df[df["Method"] == method]
    if linearmodeltree:
        model =PiecewiseRegressor(DecisionTreeRegressor(criterion='mse', max_depth=None,
                                                max_features=None,
                                                max_leaf_nodes=20,
                                                min_impurity_decrease=0.0,
                                                min_impurity_split=None,
                                                min_samples_leaf=300,
                                                min_samples_split=2,
                                                min_weight_fraction_leaf=0.0,
                                                presort=False,
                                                random_state=None,
                                                splitter='best'),
                   estimator=LinearRegression(copy_X=True, fit_intercept=True,
                                              n_jobs=None, normalize=False),
                   n_jobs=None, verbose=True)
    else:
        model = DecisionTreeRegressor(splitter="best", max_leaf_nodes=40, criterion="friedman_mse")
    # model = LinearRegression()
    dataset_features = [f for f in [
        "min_intersection", "average_intersection", "average_asjp_edit_distance", "mean_word_length", "mean_synonyms", "quantile_for_two_segments", "quantile_for_three_segments", "quantile_for_five_segments", "quantile_for_nine_segments", "n_lects", "mean_segments", "mean_asjp_segments", "mean_asjp_segments_product"]
    if f in data.columns]
    features = dataset_features + ["IThreshold", "CThreshold"] + c + parameters
    x = data[features]
    y = data[measures[0]]

    d = x.copy()
    X = squish(d)
    # poly = PolynomialFeatures(interaction_only=False, include_bias=False)
    # X = poly.fit_transform(X)

    pred = model.fit(np.asarray(X), transform(y))
    print()
    print(method)
    y_pred = untransform(pred.predict(X.values))
    score = (((y - y_pred) ** 2).sum() ** 0.5 / len(y_pred))
    if score < best_model.get(method, (None, 1e12))[1]:
        best_model[method] = pred, score
    else:
        pred, _ = best_model[method]
    y_pred = untransform(pred.predict(X.values))
    print(((y - y_pred) ** 2).sum() ** 0.5 / len(y_pred))
    print(np.quantile(np.abs(y - y_pred), 0.5))
    print(np.quantile(np.abs(y - y_pred), 0.67))
    print(np.quantile(np.abs(y - y_pred), 0.9))
    print(np.quantile(np.abs(y - y_pred), 0.95))

    if linearmodeltree:
        Exporter = GraphvizPiecewiseExporter
    else:
        Exporter = TransformedExporter
    exporter = Exporter(
        feature_names=features,
        out_file=StringIO(),
        filled=True,
        **({} if linearmodeltree else dict(transform=untransform)))
    exporter.export(pred)
    dot_data = exporter.out_file.getvalue()[:-1]
    dot_data += """
    x [label="95% intervals of the parameters:\n{:s}", fillcolor="#ffffff"] ; }}
    """.format("\n".join("{:s}: {:f}–{:f}".format(name, min, max) for name, min, max in zip(features, x.min(), x.max())))
    graph = graphviz.Source(dot_data)
    graph.render(method)

    # json.dump(dict(zip(poly.get_feature_names(x.columns), pred.coef_)),
    #           open("coef.json", "w"),
    #           sort_keys=True, indent=2)

    ymax = 0
    for i in range(200):
        r = np.random.beta(0.7, 0.7, size=(len(x.max()), 1))
        r *= 1.1
        r -= 0.05
        xt = squish(x.min().values + r * (x.max() - x.min()).values)
        yt = pred.predict(xt)
        if 1 > yt.max() > ymax or ymax == 0:
            m = dict(zip(x.max().index, xt[yt.argmax()]))
            ymax = yt.max()
    print(m, ymax)


    from pathlib import Path

    ipa_flag='-i'

    for dataset, gold in [
            ("AutoCogPhylo/data/data-aa-58-200.json",) * 2,
            ("AutoCogPhylo/data/data-ie-42-208.json",) * 2,
            ("AutoCogPhylo/data/data-st-64-110.json",) * 2,
            ("AutoCogPhylo/data/data-an-45-210.json",) * 2,
            ("AutoCogPhylo/data/data-pn-67-183.json",) * 2,
            ("potential-of-cognate-detection/code/D_test_Bahnaric-200-24.json", "potential-of-cognate-detection/code/D_test_Bahnaric-200-24.cog.tsv"),
            ("potential-of-cognate-detection/code/D_test_Chinese-180-18.json", "potential-of-cognate-detection/code/D_test_Chinese-180-18.cog.tsv"),
            ("potential-of-cognate-detection/code/D_test_Huon-140-14.json", "potential-of-cognate-detection/code/D_test_Huon-140-14.cog.tsv"),
            ("potential-of-cognate-detection/code/D_test_Romance-110-43.json", "potential-of-cognate-detection/code/D_test_Romance-110-43.cog.tsv"),
            ("potential-of-cognate-detection/code/D_test_Tujia-109-5.json", "potential-of-cognate-detection/code/D_test_Tujia-109-5.cog.tsv"),
            ("potential-of-cognate-detection/code/D_test_Uralic-173-8.json", "potential-of-cognate-detection/code/D_test_Uralic-173-8.cog.tsv"),
            ("potential-of-cognate-detection/code/D_training_Austronesian-210-20.json", "potential-of-cognate-detection/code/D_training_Austronesian-210-20.cog.tsv"),
            ("potential-of-cognate-detection/code/D_training_IndoEuropean-207-20.json", "potential-of-cognate-detection/code/D_training_IndoEuropean-207-20.cog.tsv")
    ]:
        dataset = Path(dataset)
        name = dataset.stem
        codings = (Path("codings") / name)
        codings.mkdir(parents=True, exist_ok=True)
        models = (Path("models") / name)
        models.mkdir(parents=True, exist_ok=True)

        soundclass = ["asjp", "dolgo", "sca"][np.argmax([
            1 - m["Soundclass:dolgo"] - m["Soundclass:sca"],
            m["Soundclass:dolgo"],
            m["Soundclass:sca"]])]
        clustering = ["infomap", "upgma", "single", "complete"][np.argmax([
            m["Clustering:infomap"],
            m["Clustering:upgma"],
            m["Clustering:single"],
            1 - m["Clustering:infomap"] - m["Clustering:upgma"] - m["Clustering:single"]])]
        threshold = int(m["CThreshold"]*100 + 0.5)/100
        initialthreshold = int(m["IThreshold"]*100 + 0.5)/100

        if method == "pmi":
            batch_size = int(m["Batchsize"])
            alpha = int(m["Alpha"]*100 + 0.5)/100
            model=models / "pmi-{:s}/{:03d}-{:d}-{:03d}".format(
                soundclass, int(initialthreshold*100), batch_size, int(alpha*100))
            model.parent.mkdir(parents=True, exist_ok=True)
            run=codings / "{:s}-{:03d}/{:s}/{:s}".format(clustering, int(threshold*100), model.parent.name, model.name) / "wordlist.json"
            run.parent.mkdir(parents=True, exist_ok=True)
            print("""
                    tail {run:} || python -m online_cognacy_ident.commands.train pmi \\
                                        {dataset:} --dataset-type=cldf -i \\
                                        {model:} \\
                                        -s {soundclass:} \\
                                        --initial-cutoff {initialthreshold:} \\
                                        --batch-size {batch_size:} \\
                                        --alpha {alpha:} \\
                                        --time

                    tail {run:} || python -m online_cognacy_ident.commands.run \\
                                        {model:} \\
                                        {dataset:} -i \\
                                        -s {soundclass:} \\
                                        --cluster-method={clustering:} \\
                                        --cluster-threshold={threshold:} \\
                                        --output {run:}

                    python eval_cognate_clusters.py {gold:} {run:} >> calculated_scores.txt

                                """.format(**globals()),
                  file=more)
        else:
            ratio = m["LWeight"]
            # The weight ratio between language-specific and language-independent sound changes
            gop = m["GOP"]
            mode = ["overlap", "global", "local", "dialign"][np.argmax([
                m["Alignment:overlap"],
                m["Alignment:global"],
                m["Alignment:local"],
                1 - m["Alignment:overlap"] + m["Alignment:global"] + m["Alignment:local"]])]
            model=models / "lexstat-{:s}/{:d}-{:f}-{:f}-{:s}".format(
                soundclass, int(initialthreshold*100), ratio, gop, mode)
            run=codings / "{:s}-0{:d}/{:s}/{:s}".format(clustering, int(threshold*100), model.parent.name, model.name) / "wordlist"
            run.parent.mkdir(parents=True, exist_ok=True)

            print("""
                    tail {run:}.tsv || python -m pylexirumah.autocode \\
                                        {dataset:} \\
                                        {run:} \\
                                        --cluster-method={clustering:} \\
                                        --threshold={threshold:} \\
                                        --soundclass={soundclass:} \\
                                        --initial-threshold={initialthreshold:} \\
                                        --ratio={ratio:} \\
                                        --gop={gop:} \\
                                        --mode={mode:}

                    python eval_cognate_clusters.py {gold:} {run:}.tsv >> calculated_scores.txt

                                    """.format(**globals()),
                  file=more)

more.close()
