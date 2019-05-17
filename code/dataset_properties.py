import sys
import json
from pathlib import Path
from bisect import bisect
import itertools
import collections

import lingpy
import segments
from pylexirumah import get_dataset

tokenizer = segments.tokenizer.Tokenizer()
model=lingpy.data.model.Model("asjp")
def to_asjp(segments):
    return lingpy.tokens2class(segments, model, cldf=False)
def convert_to_asjp(form):
    tokens = tokenizer(form, ipa=True).split()
    return to_asjp(tokens)
scorer = collections.defaultdict(lambda: -1)
scorer.update({(c, c): 0 for c in set(model.converter.values())})
def relative_edit_distance(x, y):
    a, b, s = lingpy.align.pairwise.pw_align(x, y, scale=1, scorer=scorer)
    return -s / len(a)

try:
    stats = json.load((Path(__file__).parent / "dataset_properties.json").open())
except (FileNotFoundError, json.decoder.JSONDecodeError):
    stats = {}

def mean(x):
    return sum(x) / len(x)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action='store_true')
    parser.add_argument("dataset", nargs="+")
    args = parser.parse_args()
    for argument in args.dataset:
        if argument in stats and not args.force:
            continue
        datafile = Path(argument)
        print(datafile)
        basename = datafile.stem

        dataset = get_dataset(datafile)
        c_language = dataset["FormTable", "languageReference"].name
        c_concept = dataset["FormTable", "parameterReference"].name
        c_form = dataset["FormTable", "form"].name
        c_segments = dataset["FormTable", "segments"].name
        lects = list(set(row[c_language] for row in dataset["FormTable"].iterdicts()))

        concepts = []
        wordlengths = []
        synonyms = []
        raw_segments = {}
        for l, language in enumerate(lects):
            print(language)
            c = {}
            concepts.append(c)
            for row in dataset["FormTable"].iterdicts():
                if row[c_language] == language:
                    c.setdefault(row[c_concept], []).append(row[c_form])
                    raw_segments.setdefault(language, set()).update(row[c_segments])
                    wordlengths.append(len(row[c_segments]))
            synonyms.extend([len(f) for f in c.values()])
        wordlengths.sort()
        asjp_segments = {language: set(to_asjp(list(phonemes)))
                         for language, phonemes in raw_segments.items()}

        intersections = []
        relative_edit_distances = []
        for concepts_i, concepts_j in itertools.combinations(concepts, 2):
            intersection = set(concepts_i).intersection(concepts_j)
            intersections.append(len(intersection))
            relative_concept_edit_distances = []
            for concept in intersection:
                for i, j in itertools.product(concepts_i[concept],
                                            concepts_j[concept]):
                    i = convert_to_asjp(i)
                    j = convert_to_asjp(j)
                    relative_concept_edit_distances.append(relative_edit_distance(i, j))
            if intersection:
                relative_edit_distances.append(
                    sum(relative_concept_edit_distances)/
                    len(relative_concept_edit_distances))

        print(basename)
        stats[argument] = {
            "min_intersection": min(intersections),
            "average_intersection": 2 * mean(intersections),
            "average_asjp_edit_distance": mean(relative_edit_distances),
            "mean_word_length": mean(wordlengths),
            "mean_synonyms": mean(synonyms),
            "quantile_for_two_segments": bisect(wordlengths, 2.01) / len(wordlengths),
            "quantile_for_three_segments": bisect(wordlengths, 3.01) / len(wordlengths),
            # In LexiRumah, a morpheme has 5.25 segments in the mean, and 90% of all marked morphemes have 9 segments or fewer.
            "quantile_for_five_segments": bisect(wordlengths, 5.25) / len(wordlengths),
            "quantile_for_nine_segments": bisect(wordlengths, 9.01) / len(wordlengths),
            "n_lects": len(lects),
            "mean_segments": mean([len(x) for x in raw_segments.values()]),
            "mean_asjp_segments": mean([len(x) for x in asjp_segments.values()]),
            "mean_asjp_segments_product": mean([
                len(x) * len(y) for x, y in itertools.combinations(asjp_segments.values(), 2)]),
        }


    json.dump(stats, Path(Path(__file__).parent / "dataset_properties.json").open("w"), indent=2, sort_keys=True)
