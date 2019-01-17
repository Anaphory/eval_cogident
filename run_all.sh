
ipa_flag='-i'

ipa_datasets=(abvd bai chinese_1964 chinese_2004 ielex japanese ob_ugrian tujia)

for dataset in \
        ~/devel/online_cognacy_ident/datasets/ielex/Wordlist-metadata.json \
        lexirumah-data/cldf/Wordlist-metadata.json \
        ~/devel/online_cognacy_ident/datasets/abvd/Wordlist-metadata.json
do
    name=$(basename $(dirname $dataset))
    echo "#" $name
    mkdir -p outputs/$name
    mkdir -p models/$name
    for soundclass in asjp sca dolgo jaeger #_color
    do
        for clustering in infomap upgma single complete
        do
            for threshold in 55 25 75 35 45 65 50 60
            do
                # OnlinePMI
                for batch_size in 64 256 1024 128 512 2048
                do
                    for alpha in 0.7 0.5 0.8 0.75 0.6
                    do
                        model=pmi-$soundclass-0$threshold-$batch_size-$alpha
                        run=$clustering-$model

                        if [[ ! -f models/$name/$model ]]
                        then
                            echo python -m online_cognacy_ident.commands.train pmi \
                                --data cldf $dataset $ipa_flag \
                                models/$name/$model \
                                -s $soundclass \
                                --batch-size $batch_size --alpha $alpha \
                                --time
                        fi

                        if [[ ! -f outputs/$name/$run-metadata.json ]]
                        then
                            echo python -m online_cognacy_ident.commands.run \
                                models/$name/$model \
                                ~/databases/lexirumah-data/cldf/Wordlist-metadata.json $ipa_flag \
                                -s $soundclass \
                                --cluster-method=$clustering \
                                --output outputs/$name/$run-metadata.json
                        fi
                    done
                done

                # LexStat
                for ratio in 0.5 1 3 10 0 # The weight ratio between language-specific and language-independent sound changes
                do
                    run=lexstat-$soundclass-$clustering-0$threshold-$ratio

                    echo python -m pylexirumah.autocode \
                        $dataset\
                        outputs/$name/$run \
                        --threshold=0.$threshold \
                        --cluster-method=$clustering \
                        --soundclass=$soundclass \
                        --ratio=$ratio
                done
            done
        done
    done
done

