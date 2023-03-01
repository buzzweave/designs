import argparse
import json

import matplotlib.pyplot as plt


def main(args):
    results = {}
    if args.conll_03_path:
        with open(args.conll_03_path, "r") as f:
            results["conll_03"] = json.load(f)
    if args.wnut_17_path:
        with open(args.wnut_17_path, "r") as f:
            results["wnut_17"] = json.load(f)
    if args.ontonotes_path:
        with open(args.ontonotes_path, "r") as f:
            results["ontonotes"] = json.load(f)
    if args.fewnerdfine_path:
        with open(args.fewnerdfine_path, "r") as f:
            results["fewnerd_fine"] = json.load(f)
    if args.fewnerdcoarse_path:
        with open(args.fewnerdcoarse_path, "r") as f:
            results["fewnerd_coarse"] = json.load(f)

    if not results:
        raise Exception("No results.")

    for k in args.k:
        for corpus in ["conll_03", "wnut_17", "ontonotes", "fewnerd_coarse", "fewnerd_fine"]:
            plt.plot(
                ["1", "2", "4", "8", "16"],
                [4.3, 15.7, 51.4, 67.5, 78.9],
                color="tab:blue",
                marker="o",
                linestyle="-",
                label="FLERT",
            )
            plt.plot(
                ["1", "2", "4", "8", "16"],
                [10.8, 16.6, 29.0, 52.9, 62.7],
                color="tab:orange",
                marker="o",
                linestyle="-",
                label="Dual Encoder",
            )
            plt.legend(loc="lower right")
            plt.xlabel("k-shot")
            plt.ylabel("f1-score")
            plt.title("Low Resource on CoNLL03")
            plt.savefig("lowresource_conll.png")
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conll_03_path", type=str, default="")
    parser.add_argument("--wnut_17_path", type=str, default="")
    parser.add_argument("--ontonotes_path", type=str, default="")
    parser.add_argument("--fewnerdfine_path", type=str, default="")
    parser.add_argument("--fewnerdcoarse_path", type=str, default="")
    parser.add_argument("--k", type=int, default=1, nargs="+")
    args = parser.parse_args()
    main(args)
