import argparse
import os

import numpy as np

import flair
from flair.datasets import CONLL_03, ONTONOTES, WNUT_17, ColumnCorpus
from flair.models import DualEncoder
from flair.trainers import ModelTrainer


def main(args):
    if args.cuda:
        flair.device = f"cuda:{args.cuda_device}"

    average_result = []
    if args.k > 0:
        for support_set_id in range(5):
            if args.corpus == "wnut_17":
                few_shot_corpus = ColumnCorpus(
                    data_folder=f"data/fewshot/wnut17/{args.k}shot/",
                    train_file=f"{support_set_id}.txt",
                    column_format={0: "text", 1: "ner"},
                    sample_missing_splits=False,
                    label_name_map={
                        "corporation": "corporation",
                        "creative-work": "creative work",
                        "group": "group",
                        "location": "location",
                        "person": "person",
                        "product": "product",
                    },
                )

                full_corpus = WNUT_17(
                    label_name_map={
                        "corporation": "corporation",
                        "creative-work": "creative work",
                        "group": "group",
                        "location": "location",
                        "person": "person",
                        "product": "product",
                    }
                )
            elif args.corpus == "conll_03":
                few_shot_corpus = ColumnCorpus(
                    data_folder=f"data/fewshot/conll03/{args.k}shot/",
                    train_file=f"{support_set_id}.txt",
                    column_format={0: "text", 1: "ner"},
                    sample_missing_splits=False,
                    label_name_map={"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "miscellaneous"},
                )

                full_corpus = CONLL_03(
                    base_path="data",
                    column_format={0: "text", 1: "pos", 2: "chunk", 3: "ner"},
                    label_name_map={"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "miscellaneous"},
                )
            elif args.corpus == "ontonotes":
                few_shot_corpus = ONTONOTES(
                    label_name_map={
                        "CARDINAL": "cardinal",
                        "DATE": "date",
                        "EVENT": "event",
                        "FAC": "facility",
                        "GPE": "geographical social political entity",
                        "LANGUAGE": "language",
                        "LAW": "law",
                        "LOC": "location",
                        "MONEY": "money",
                        "NORP": "nationality religion political",
                        "ORDINAL": "ordinal",
                        "ORG": "organization",
                        "PERCENT": "percent",
                        "PERSON": "person",
                        "PRODUCT": "product",
                        "QUANTITY": "quantity",
                        "TIME": "time",
                        "WORK_OF_ART": "work of art",
                    }
                ).to_nway_kshot(n=-1, k=args.k, tag_type="ner", seed=support_set_id, include_validation=False)
                full_corpus = few_shot_corpus
            else:
                raise Exception("no valid corpus.")

            tag_type = "ner"
            label_dictionary = few_shot_corpus.make_label_dictionary(tag_type, add_unk=False)
            # force spans == true, there is one split containing only B-*'s
            label_dictionary.span_labels = True

            pretrained_model_path = f"{args.cache_path}/pretrained-dual-encoder/{args.transformer}_{args.pretraining_corpus}{args.fewnerd_granularity}_{args.lr}_{args.seed}/final-model.pt"
            model = DualEncoder.load(pretrained_model_path)
            model._init_verbalizers_and_tag_dictionary(tag_dictionary=label_dictionary)

            trainer = ModelTrainer(model, few_shot_corpus)

            save_path = (
                f"{args.cache_path}/fewshot-dual-encoder/"
                f"{args.transformer}_{args.corpus}_{args.lr}_{args.seed}_pretrained_on{args.pretraining_corpus}{args.fewnerd_granularity}/"
                f"{args.k}shot_{support_set_id}"
            )

            trainer.fine_tune(
                save_path,
                learning_rate=args.lr,
                mini_batch_size=args.bs,
                mini_batch_chunk_size=args.mbs,
                max_epochs=args.epochs,
                save_final_model=False,
            )

            result = model.evaluate(
                data_points=full_corpus.test,
                gold_label_type=tag_type,
                out_path=f"{save_path}/predictions.txt",
            )
            with open(
                f"{save_path}/result.txt",
                "w",
            ) as f:
                f.writelines(result.detailed_results)

            average_result.append(result.main_score)

        average_result = [round(float(score) * 100, 2) for score in average_result]
        with open(
            f"{args.cache_path}/fewshot-dual-encoder/"
            f"{args.transformer}_{args.corpus}_{args.lr}_{args.seed}_pretrained_on{args.pretraining_corpus}{args.fewnerd_granularity}/"
            f"{args.k}shot.txt",
            "w",
        ) as f:
            f.write(f"all results: {average_result} \n")
            f.write(f"average: {np.mean(average_result)} \n")
            f.write(f"std: {np.std(average_result)} \n")
    elif args.k == 0:
        if args.corpus == "wnut_17":
            full_corpus = WNUT_17(
                label_name_map={
                    "corporation": "corporation",
                    "creative-work": "creative work",
                    "group": "group",
                    "location": "location",
                    "person": "person",
                    "product": "product",
                }
            )
        elif args.corpus == "conll_03":
            full_corpus = CONLL_03(
                base_path="data",
                column_format={0: "text", 1: "pos", 2: "chunk", 3: "ner"},
                label_name_map={"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "miscellaneous"},
            )
        elif args.corpus == "ontonotes":
            full_corpus = ONTONOTES(
                label_name_map={
                    "CARDINAL": "cardinal",
                    "DATE": "date",
                    "EVENT": "event",
                    "FAC": "facility",
                    "GPE": "geographical social political entity",
                    "LANGUAGE": "language",
                    "LAW": "law",
                    "LOC": "location",
                    "MONEY": "money",
                    "NORP": "nationality religion political",
                    "ORDINAL": "ordinal",
                    "ORG": "organization",
                    "PERCENT": "percent",
                    "PERSON": "person",
                    "PRODUCT": "product",
                    "QUANTITY": "quantity",
                    "TIME": "time",
                    "WORK_OF_ART": "work of art",
                }
            )
        else:
            raise Exception("no valid corpus.")

        tag_type = "ner"
        label_dictionary = full_corpus.make_label_dictionary(tag_type, add_unk=False)
        # force spans == true, there is one split containing only B-*'s
        label_dictionary.span_labels = True

        pretrained_model_path = f"{args.cache_path}/pretrained-dual-encoder/{args.transformer}_{args.pretraining_corpus}{args.fewnerd_granularity}_{args.lr}_{args.seed}/final-model.pt"
        model = DualEncoder.load(pretrained_model_path)
        model._init_verbalizers_and_tag_dictionary(tag_dictionary=label_dictionary)

        save_path = (
            f"{args.cache_path}/fewshot-dual-encoder/"
            f"{args.transformer}_{args.corpus}_{args.lr}_{args.seed}_pretrained_on{args.pretraining_corpus}{args.fewnerd_granularity}/"
            f"0shot"
        )

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        result = model.evaluate(
            data_points=full_corpus.test,
            gold_label_type=tag_type,
            out_path=f"{save_path}/predictions.txt",
        )
        with open(
            f"{save_path}/result.txt",
            "w",
        ) as f:
            f.writelines(result.detailed_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--cuda_device", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cache_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon/flair-models")
    parser.add_argument("--corpus", type=str, default="conll_03")
    parser.add_argument("--pretraining_corpus", type=str, default="ontonotes")
    parser.add_argument("--fewnerd_granularity", type=str, default="")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--transformer", type=str, default="bert-base-cased")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    main(args)
