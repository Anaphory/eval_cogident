#!/usr/bin/env python

"""Compare cognates in a CLDF Wordlist with a gold standard"""

import csv
from pycldf.util import Path
import argparse

import bcubed
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix

from pylexirumah.util import get_dataset, cognate_sets


def pprint_form(form_id):
    print("{:8} {:20s} {:20s} {:s}".format(
        forms[form_id][c_id],
        forms[form_id][c_lect],
        forms[form_id][c_concept],
        " ".join(forms[form_id][c_segm])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("gold",
                        type=Path,
                        help="A CLDF dataset with ground-truth cognate codes")
    parser.add_argument("codings",
                        type=Path,
                        help="A CLDF dataset with cognate codes")
    parser.add_argument("--gold-lingpy", action="store_true",
                        default=False,
                        help="The ground-truth data is in LingPy's format, not CLDF.")
#     parser.add_argument("--lingpy", action="store_true",
#                         default=False,
#                         help="The data is in LingPy's format, not CLDF.")
    parser.add_argument("--ssv", default=False,
                        action="store_true",
                        help="Output one line, not many")
    args = parser.parse_args()

    if args.codings.suffix == '.tsv':
        # Assume LingPy
        import lingpy
        dataset = lingpy.LexStat(str(args.codings))
        forms = {row:
            {e: dataset[row][dataset.header[e]]
             for e in dataset.entries
             if e in dataset.header}
            for row in dataset}
        codings = {
            str(form): str(row["cogid"])
            for form, row in forms.items()}

        def iterate_concept_and_id():
            for i in dataset:
                yield dataset[i][dataset.header['concept']], str(i)
    else:
        dataset = get_dataset(args.codings)
        cognatesets = cognate_sets(dataset)
        codings = {
            str(form): code
            for code, forms in cognatesets.items()
            for form in forms}

        c_concept = dataset["FormTable", "parameterReference"].name
        c_id = dataset["FormTable", "id"].name

        def iterate_concept_and_id():
            for row in dataset["FormTable"].iterdicts():
                yield row[c_concept], str(row[c_id])


    if args.gold.suffix == ".tsv":
        import lingpy
        gold_dataset = lingpy.LexStat(str(args.gold))
        gold_forms = {row:
            {e: gold_dataset[row][gold_dataset.header[e]]
             for e in gold_dataset.entries
             if e in gold_dataset.header}
            for row in gold_dataset}
        gold_codings = {
            str(form): str(row["cogid"])
            for form, row in gold_forms.items()}
    else:
        gold_dataset = get_dataset(args.gold)
        gold_cognatesets = cognate_sets(gold_dataset, code_column="COGID")
        gold_codings = {
            str(form): code
            for code, forms in gold_cognatesets.items()
            for form in forms}


    concept_codes = {}
    for concept, id in iterate_concept_and_id():
        gold_c, c = concept_codes.setdefault(concept, ([], []))
        gold_c.append(''.join([str(s) for s in gold_codings.get(id, ())]))
        c.append(''.join([str(s) for s in codings.get(id, ())]))

    v = 0
    r = 0
    a = 0
    b = 0
    for concept, (gold_c, c) in concept_codes.items():
        v += metrics.v_measure_score(gold_c, c)
        r += metrics.adjusted_rand_score(gold_c, c)
        a += metrics.adjusted_mutual_info_score(gold_c, c)
        b += bcubed.fscore(bcubed.simple_precision(c, gold_c),
                           bcubed.simple_recall(c, gold_c))
    norm = len(concept_codes)
    print(args.codings, b/norm, v/norm, r/norm, a/norm)
