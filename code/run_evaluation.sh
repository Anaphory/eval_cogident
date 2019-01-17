(for d in ielex abvd; do ls outputs/$d/*.{json,tsv} | xargs -n1 python eval_cognate_clusters.py ~/devel/online_cognacy_ident/datasets/$d/Wordlist-metadata.json 2> /dev/null ; done) > scores.txt
