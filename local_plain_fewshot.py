import argparse
from pathlib import Path

import numpy as np

import flair
from flair.datasets import CONLL_03, FEWNERD, ONTONOTES, WNUT_17
from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings
from flair.models import DualEncoder
from flair.optim import LinearSchedulerWithWarmup
from flair.trainers import ModelTrainer
from flair.training_utils import AnnealOnPlateau


def main(args):
    if args.cuda:
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = Path(
        f"{args.cache_path}/fewshot-dual-encoder/"
        f"{args.transformer}_{args.corpus}{args.fewnerd_granularity}_{args.lr}_{args.seed}/"
    )

    for k in args.k:
        average_result = []
        for seed in range(5):
            if args.corpus == "wnut_17":
                corpus = WNUT_17(
                    label_name_map={
                        "corporation": "corporation",
                        "creative-work": "creative work",
                        "group": "group",
                        "location": "location",
                        "person": "person",
                        "product": "product",
                    }
                ).to_nway_kshot(n=-1, k=k, tag_type="ner", seed=seed, include_validation=False)
            elif args.corpus == "conll_03":
                corpus = CONLL_03(
                    base_path="data",
                    column_format={0: "text", 1: "pos", 2: "chunk", 3: "ner"},
                    label_name_map={"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "miscellaneous"},
                ).to_nway_kshot(n=-1, k=k, tag_type="ner", seed=seed, include_validation=False)
            elif args.corpus == "ontonotes":
                corpus = ONTONOTES(
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
                ).to_nway_kshot(n=-1, k=k, tag_type="ner", seed=seed, include_validation=False)
            elif args.corpus == "fewnerd":
                if args.fewnerd_granularity == "fine":
                    corpus = FEWNERD(
                        label_name_map={
                            "location-GPE": "geographical social political entity",
                            "person-other": "other person",
                            "organization-other": "other organization",
                            "organization-company": "company",
                            "person-artist/author": "author artist",
                            "person-athlete": "athlete",
                            "person-politician": "politician",
                            "building-other": "other building",
                            "organization-sportsteam": "sportsteam",
                            "organization-education": "eduction",
                            "location-other": "other location",
                            "other-biologything": "biology",
                            "location-road/railway/highway/transit": "road railway highway transit",
                            "person-actor": "actor",
                            "prodcut-other": "other product",
                            "event-sportsevent": "sportsevent",
                            "organization-government/governmentagency": "government agency",
                            "location-bodiesofwater": "bodies of water",
                            "organization-media/newspaper": "media newspaper",
                            "art-music": "music",
                            "other-chemicalthing": "chemical",
                            "event-attack/battle/war/militaryconflict": "attack war battle military conflict",
                            "organization-politicalparty": "political party",
                            "art-writtenart": "written art",
                            "other-award": "award",
                            "other-livingthing": "living thing",
                            "event-other": "other event",
                            "art-film": "film",
                            "product-software": "software",
                            "organization-sportsleague": "sportsleague",
                            "other-language": "language",
                            "other-disease": "disease",
                            "organization-showorganization": "show organization",
                            "product-airplane": "airplane",
                            "other-astronomything": "astronomy",
                            "organization-religion": "religion",
                            "product-car": "car",
                            "person-scholar": "scholar",
                            "other-currency": "currency",
                            "person-soldier": "soldier",
                            "location-mountain": "mountain",
                            "art-broadcastprogram": "broadcastprogram",
                            "location-island": "island",
                            "art-other": "other art",
                            "person-director": "director",
                            "product-weapon": "weapon",
                            "other-god": "god",
                            "building-theater": "theater",
                            "other-law": "law",
                            "product-food": "food",
                            "other-medical": "medical",
                            "product-game": "game",
                            "location-park": "park",
                            "product-ship": "ship",
                            "building-sportsfacility": "sportsfacility",
                            "other-educationaldegree": "educational degree",
                            "building-airport": "airport",
                            "building-hospital": "hospital",
                            "product-train": "train",
                            "building-library": "library",
                            "building-hotel": "hotel",
                            "building-restaurant": "restaurant",
                            "event-disaster": "disaster",
                            "event-election": "election",
                            "event-protest": "protest",
                            "art-painting": "painting",
                        }
                    ).to_nway_kshot(n=-1, k=k, tag_type="ner", seed=seed, include_validation=False)
                elif args.fewnerd_granularity == "coarse":
                    corpus = FEWNERD(
                        label_name_map={
                            "location-GPE": "location",
                            "person-other": "person",
                            "organization-other": "organization",
                            "organization-company": "organization",
                            "person-artist/author": "person",
                            "person-athlete": "person",
                            "person-politician": "person",
                            "building-other": "building",
                            "organization-sportsteam": "organization",
                            "organization-education": "organization",
                            "location-other": "location",
                            "other-biologything": "biology",
                            "location-road/railway/highway/transit": "location",
                            "person-actor": "person",
                            "prodcut-other": "product",
                            "event-sportsevent": "event",
                            "organization-government/governmentagency": "organization",
                            "location-bodiesofwater": "location",
                            "organization-media/newspaper": "organization",
                            "art-music": "art",
                            "other-chemicalthing": "chemical",
                            "event-attack/battle/war/militaryconflict": "event",
                            "organization-politicalparty": "organization",
                            "art-writtenart": "art",
                            "other-award": "award",
                            "other-livingthing": "living thing",
                            "event-other": "event",
                            "art-film": "art",
                            "product-software": "product",
                            "organization-sportsleague": "organization",
                            "other-language": "language",
                            "other-disease": "disease",
                            "organization-showorganization": "organization",
                            "product-airplane": "product",
                            "other-astronomything": "astronomy",
                            "organization-religion": "organization",
                            "product-car": "product",
                            "person-scholar": "person",
                            "other-currency": "currency",
                            "person-soldier": "person",
                            "location-mountain": "location",
                            "art-broadcastprogram": "art",
                            "location-island": "location",
                            "art-other": "art",
                            "person-director": "person",
                            "product-weapon": "product",
                            "other-god": "god",
                            "building-theater": "building",
                            "other-law": "law",
                            "product-food": "product",
                            "other-medical": "medical",
                            "product-game": "product",
                            "location-park": "location",
                            "product-ship": "product",
                            "building-sportsfacility": "building",
                            "other-educationaldegree": "educational degree",
                            "building-airport": "building",
                            "building-hospital": "building",
                            "product-train": "product",
                            "building-library": "building",
                            "building-hotel": "building",
                            "building-restaurant": "building",
                            "event-disaster": "event",
                            "event-election": "event",
                            "event-protest": "event",
                            "art-painting": "art",
                        }
                    ).to_nway_kshot(n=-1, k=k, tag_type="ner", seed=seed, include_validation=False)
                elif args.fewnerd_granularity == "coarse-fine":
                    corpus = FEWNERD(
                        label_name_map={
                            "location-GPE": "location geographical social political entity",
                            "person-other": "person other",
                            "organization-other": "organization other",
                            "organization-company": "organization company",
                            "person-artist/author": "person author artist",
                            "person-athlete": "person athlete",
                            "person-politician": "person politician",
                            "building-other": "building other",
                            "organization-sportsteam": "organization sportsteam",
                            "organization-education": "organization eduction",
                            "location-other": "location other",
                            "other-biologything": "other biology",
                            "location-road/railway/highway/transit": "location road railway highway transit",
                            "person-actor": "person actor",
                            "prodcut-other": "product other",
                            "event-sportsevent": "event sportsevent",
                            "organization-government/governmentagency": "organization government agency",
                            "location-bodiesofwater": "location bodies of water",
                            "organization-media/newspaper": "organization media newspaper",
                            "art-music": "art music",
                            "other-chemicalthing": "other chemical",
                            "event-attack/battle/war/militaryconflict": "event attack war battle military conflict",
                            "organization-politicalparty": "organization political party",
                            "art-writtenart": "art written art",
                            "other-award": "other award",
                            "other-livingthing": "other living",
                            "event-other": "event other",
                            "art-film": "art film",
                            "product-software": "product software",
                            "organization-sportsleague": "organization sportsleague",
                            "other-language": "other language",
                            "other-disease": "other disease",
                            "organization-showorganization": "organization show organization",
                            "product-airplane": "product airplane",
                            "other-astronomything": "other astronomy",
                            "organization-religion": "organization religion",
                            "product-car": "product car",
                            "person-scholar": "person scholar",
                            "other-currency": "other currency",
                            "person-soldier": "person soldier",
                            "location-mountain": "location mountain",
                            "art-broadcastprogram": "art broadcastprogram",
                            "location-island": "location island",
                            "art-other": "art other",
                            "person-director": "person director",
                            "product-weapon": "product weapon",
                            "other-god": "other god",
                            "building-theater": "building theater",
                            "other-law": "other law",
                            "product-food": "product food",
                            "other-medical": "other medical",
                            "product-game": "product game",
                            "location-park": "location park",
                            "product-ship": "product ship",
                            "building-sportsfacility": "building sportsfacility",
                            "other-educationaldegree": "other educational degree",
                            "building-airport": "building airport",
                            "building-hospital": "building hospital",
                            "product-train": "product train",
                            "building-library": "building library",
                            "building-hotel": "building hotel",
                            "building-restaurant": "building restaurant",
                            "event-disaster": "event disaster",
                            "event-election": "event election",
                            "event-protest": "event protest",
                            "art-painting": "art painting",
                        }
                    ).to_nway_kshot(n=-1, k=k, tag_type="ner", seed=seed, include_validation=False)
            else:
                raise Exception("no valid corpus.")

            tag_type = "ner"
            label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)
            label_dictionary.span_labels = True

            token_encoder = TransformerWordEmbeddings(args.transformer)
            label_encoder = TransformerDocumentEmbeddings(args.transformer)

            model = DualEncoder(
                token_encoder=token_encoder,
                label_encoder=label_encoder,
                tag_dictionary=label_dictionary,
                tag_type=tag_type,
            )

            trainer = ModelTrainer(model, corpus)

            save_path = save_base_path / f"{k}shot_{seed}"

            trainer.fine_tune(
                save_path,
                learning_rate=args.lr,
                mini_batch_size=args.bs,
                mini_batch_chunk_size=args.mbs,
                max_epochs=args.epochs,
                scheduler=AnnealOnPlateau if args.early_stopping else LinearSchedulerWithWarmup,
                train_with_dev=args.early_stopping,
                min_learning_rate=args.min_lr if args.early_stopping else 0.001,
                save_final_model=False,
            )

            result = model.evaluate(
                data_points=corpus.test,
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
            save_base_path / f"{k}shot_results.txt",
            "w",
        ) as f:
            f.write(f"all results: {average_result} \n")
            f.write(f"average: {np.mean(average_result)} \n")
            f.write(f"std: {np.std(average_result)} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--cuda_device", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cache_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon/flair-models")
    parser.add_argument("--corpus", type=str, default="conll_03")
    parser.add_argument("--fewnerd_granularity", type=str, default="")
    parser.add_argument("--k", type=list, default=[1])
    parser.add_argument("--transformer", type=str, default="bert-base-uncased")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    args = parser.parse_args()
    main(args)
