import argparse

import flair
from flair.models import FewshotClassifier, TARSTagger
from flair.trainers import ModelTrainer
from local_domain_extension_label_name_map import get_corpus


def main(args):
    flair.set_seed(args.seed)

    if args.cuda:
        flair.device = f"cuda:{args.cuda_device}"

    tars_tagger: FewshotClassifier = TARSTagger(
        embeddings=args.transformer,
        num_negative_labels_to_sample=1,
        prefix=True,
    )

    corpus = get_corpus(args.corpus, map=args.label_map_type)
    print(corpus)

    dictionary = corpus.make_label_dictionary("ner")
    print(dictionary)

    tars_tagger.add_and_switch_to_new_task(
        task_name="pretraining", label_dictionary=dictionary, label_type="ner", force_switch=True
    )

    trainer = ModelTrainer(tars_tagger, corpus)

    trainer.fine_tune(
        f"{args.cache_path}/flair-models/pretrained-tart/{args.transformer}_{args.corpus}_{args.label_map_type}_{args.lr}_{args.seed}",
        learning_rate=args.lr,
        mini_batch_size=args.bs,
        mini_batch_chunk_size=args.mbs,
        train_with_dev=True,
        monitor_test=args.monitor_test,
        save_model_each_k_epochs=args.save_every_n_epochs,
        max_epochs=args.epochs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cache_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon")
    parser.add_argument("--corpus", type=str, default="fwenerd")
    parser.add_argument("--transformer", type=str, default="bert-base-uncased")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=2)
    parser.add_argument("--monitor_test", type=bool, default=False)
    parser.add_argument("--label_map_type", type=str, default="coarse-fine")
    parser.add_argument("--save_every_n_epochs", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
