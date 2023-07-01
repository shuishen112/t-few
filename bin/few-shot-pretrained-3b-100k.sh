for dataset in copa h-swag storycloze winogrande wsc wic rte cb anli-r1 anli-r2 anli-r3
do
    for seed in 42 1024 0 1 32
    do
        python -m src.pl_train -c t03b.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/t03b_ia3_finish.pt" exp_name=t03b_${dataset}_seed${seed}_ia3_pretrained100k few_shot_random_seed=${seed} seed=${seed}
    done
done

# python -m src.pl_train -c t03b.json+ia3_emb2ket.json+copa.json -k load_weight="t03b_pretrain_emb2ket/global_step80000.pt" exp_name=t03b_copa_seed_1024__pretrained100k few_shot_random_seed=1024 seed=1024 allow_skip_exp=False
