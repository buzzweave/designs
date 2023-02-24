import argparse
from pathlib import Path

import numpy as np

import flair
from flair.datasets import ONTONOTES, ColumnCorpus
from flair.models import FewshotClassifier, TARSTagger
from flair.trainers import ModelTrainer
from local_domain_extension_label_name_map import get_corpus, get_label_name_map


def main(args):
    flair.set_seed(args.seed)

    if args.cuda:
        flair.device = f"cuda:{args.cuda_device}"

    full_corpus = get_corpus(name=args.corpus, map=args.label_map_type)
    average_over_support_sets = []
    base_save_path = Path(
        f"{args.cache_path}/flair-models/fewshot-tart/"
        f"{args.transformer}_{args.corpus}_{args.lr}_{args.seed}"
        f"_pretrained_on_{args.pretraining_corpus}{f'_{args.fewnerd_granularity}' if args.fewnerd_granularity != '' else ''}"
        f"/{args.k}shot/"
    )
    for split in range(args.splits):
        target_path = (
            f"{args.cache_path}/flair-models/pretrained-tart/"
            f"{args.transformer}_{args.pretraining_corpus}{f'_{args.fewnerd_granularity}' if args.fewnerd_granularity != '' else ''}"
            f"_{args.lr}-{args.seed}/final-model.pt"
        )
        try:
            tars_tagger: FewshotClassifier = TARSTagger.load(target_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"{target_path} - has this model been trained?")

        if args.corpus == "ontonotes":
            support_set = ONTONOTES(label_name_map=get_label_name_map(args.corpus)).to_nway_kshot(
                n=-1, k=args.k, tag_type="ner", seed=split, include_validation=False
            )
        else:
            support_set = ColumnCorpus(
                data_folder=f"data/fewshot/{args.corpus}/{args.k}shot",
                train_file=f"{split}.txt",
                sample_missing_splits=False,
                column_format={0: "text", 1: "ner"},
                label_name_map=get_label_name_map(args.corpus),
            )
        print(support_set)

        dictionary = support_set.make_label_dictionary("ner")
        print(dictionary)

        tars_tagger.add_and_switch_to_new_task(
            task_name="fewshot", label_dictionary=dictionary, label_type="ner", force_switch=True
        )

        trainer = ModelTrainer(tars_tagger, support_set)

        trainer.fine_tune(
            base_save_path / f"split_{split}",
            learning_rate=args.lr,
            mini_batch_size=args.bs,
            mini_batch_chunk_size=args.mbs,
            save_final_model=False,
            max_epochs=args.epochs,
        )

        result = tars_tagger.evaluate(
            data_points=full_corpus.test,
            gold_label_type="ner",
            out_path=f"{base_save_path / f'split_{split}'}/predictions.txt",
        )
        with open(f"{base_save_path / f'split_{split}'}/result.txt", "w") as f:
            f.write(result.detailed_results)

        average_over_support_sets.append(result.main_score)

    with open(
        f"{base_save_path}/average_result.txt",
        "w",
    ) as f:
        results = [round(float(x) * 100, 2) for x in average_over_support_sets]
        f.write(f"scores: {results}")
        f.write(f"average micro f1: {np.mean(results)}")
        f.write(f"std micro f1: {np.std(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cache_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon")
    parser.add_argument("--pretraining_corpus", type=str, default="ontonotes")
    parser.add_argument("--corpus", type=str, default="conll03")
    parser.add_argument("--label_map_type", type=str, default="short")
    parser.add_argument("--transformer", type=str, default="bert-base-uncased")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stopping", type=bool, default=False)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--pretraining_lr", type=float, default=1e-5)
    parser.add_argument("--fewnerd_granularity", type=str, default="")
    args = parser.parse_args()
    main(args)
