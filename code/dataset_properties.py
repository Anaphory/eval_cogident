import sys
import json
from pathlib import Path
import itertools
import collections

import lingpy
import segments
from pylexirumah import get_dataset

tokenizer = segments.tokenizer.Tokenizer()
model=lingpy.data.model.Model("asjp")
def convert_to_asjp(form):
    tokens = tokenizer(form, ipa=True).split()
    return lingpy.tokens2class(tokens, model, cldf=False)
scorer = collections.defaultdict(lambda: -1)
scorer.update({(c, c): 0 for c in set(model.converter.values())})
def relative_edit_distance(x, y):
    a, b, s = lingpy.align.pairwise.pw_align(x, y, scale=1, scorer=scorer)
    return -s / len(a)

try:
    stats = json.load((Path(__file__).parent / "dataset_properties.json").open())
except (FileNotFoundError, json.decoder.JSONDecodeError):
    stats = {}

if __name__ == "__main__":
    for argument in sys.argv[1:]:
        if argument in stats:
            continue
        datafile = Path(argument)
        print(datafile)
        basename = datafile.stem

        dataset = get_dataset(datafile)
        c_language = dataset["FormTable", "languageReference"].name
        c_concept = dataset["FormTable", "parameterReference"].name
        c_form = dataset["FormTable", "form"].name
        lects = list(set(row[c_language] for row in dataset["FormTable"].iterdicts()))

        concepts = []
        for l, language in enumerate(lects):
            print(language)
            c = {}
            concepts.append(c)
            for row in dataset["FormTable"].iterdicts():
                if row[c_language] == language:
                    c.setdefault(row[c_concept], []).append(row[c_form])

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
            "average_intersection": 2 * sum(intersections) / len(intersections),
            "average_asjp_edit_distance": sum(relative_edit_distances) / len(relative_edit_distances),
            "n_lects": len(lects),
        }


    json.dump(stats, Path(Path(__file__).parent / "dataset_properties.json").open("w"), indent=2)
