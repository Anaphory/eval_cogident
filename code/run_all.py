from pathlib import Path

ipa_flag='-i'

deviations = [[] for _ in range(20)]

with open("calculated_scores.txt", "w") as _:
    pass
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
    for soundclass, ic in [("asjp", 0), ("sca", 0), ("dolgo", 1), ("jaeger", 1), ("_color", 2)]:
        for clustering, im in [("infomap", 0), ("upgma", 1), ("single", 2), ("complete", 2)]:
            for threshold, it in [(55, 0), (25, 1), (75, 1), (35, 1), (45, 1), (65, 1), (50, 1), (60, 1)]:
                for initialthreshold, ii in [(70, 0), (50, 0), (40, 1), (90, 1), (80, 1), (30, 1), (5, 2), (99, 2)]:
                    # OnlinePMI
                    for batch_size, ib in [(64, 0), (256, 1), (1024, 1), (128, 1), (512, 1), (2048, 1)]:
                        for alpha, ia in [(70, 0), (50, 1), (80, 1), (75, 1), (60, 1)]:
                            model=models / "pmi-{:s}/0{:d}-{:d}-{:d}".format(
                                soundclass, initialthreshold, batch_size, alpha)
                            model.parent.mkdir(parents=True, exist_ok=True)
                            run=codings / "{:s}-0{:d}/{:s}/{:s}".format(clustering, threshold, model.parent.name, model.name) / "wordlist.json"
                            run.parent.mkdir(parents=True, exist_ok=True)
                            deviations[ic+im+it+ii+ib+ia].append("""
                tail {run:} || python -m online_cognacy_ident.commands.train pmi \\
                                    {dataset:} --dataset-type=cldf -i \\
                                    {model:} \\
                                    -s {soundclass:} \\
                                     --initial-cutoff 0.{initialthreshold:} \\
                                    --batch-size {batch_size:} \\
                                    --alpha 0.{alpha:} \\
                                    --time

                tail {run:} || python -m online_cognacy_ident.commands.run \\
                                    {model:} \\
                                    {dataset:} -i \\
                                    -s {soundclass:} \\
                                    --cluster-method={clustering:} \\
                                    --cluster-threshold=0.{threshold:} \\
                                    --output {run:}

                python eval_cognate_clusters.py {gold:} {run:} >> calculated_scores.txt

                            """.format(**globals()))
                    # LexStat
                    for ratio, ir in [(0.5, 1), (1, 1), (3, 0), (10, 1), (0, 1)]:
                        # The weight ratio between language-specific and
                        # language-independent sound changes
                        for gop, ig in [(-2, 0), (-1.5, 1), (-2.5, 1), (-4, 1), (-0.5, 2)]:
                            for mode, ia in [("overlap", 0), ("global", 1), ("local", 1), ("dialign", 1)]:
                                model=models / "lexstat-{:s}/0{:d}-{:f}-{:f}-{:s}".format(
                                    soundclass, initialthreshold, ratio, gop, mode)
                                run=codings / "{:s}-0{:d}/{:s}/{:s}".format(clustering, threshold, model.parent.name, model.name) / "wordlist"
                                run.parent.mkdir(parents=True, exist_ok=True)

                                deviations[ic+im+it+ii+ir+ig+ia].append("""
                tail {run:}.tsv || python -m pylexirumah.autocode \\
                                    {dataset:} \\
                                    {run:} \\
                                    --cluster-method={clustering:} \\
                                    --threshold=0.{threshold:} \\
                                    --soundclass={soundclass:} \\
                                    --initial-threshold=0.{initialthreshold:} \\
                                    --ratio={ratio:} \\
                                    --gop={gop:} \\
                                    --mode={mode:}

                python eval_cognate_clusters.py {gold:} {run:}.tsv >> calculated_scores.txt

                                """.format(**globals()))

for b, block in enumerate(deviations):
    print("#", b)
    for i in block:
        print(i)
