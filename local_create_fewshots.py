import json

from torch.utils.data.dataset import Subset

from flair.datasets import CONLL_03, FEWNERD, ONTONOTES, WNUT_17


def main():
    conll()
    wnut()
    ontonotes()
    fewnerd_fine()
    fewnerd_coarse()
    print()


def conll():
    conll_indices = {}
    conll = CONLL_03(base_path="data")

    no_docstarts = []
    for idx, sentence in enumerate(conll.train):
        if "DOCSTART" in sentence.text:
            pass
        else:
            no_docstarts.append(idx)
    conll._train = Subset(conll._train, no_docstarts)

    tag_type = "ner"
    label_dict = conll.make_label_dictionary(tag_type, add_unk=False)
    labels = [label.decode("utf-8") for label in label_dict.idx2item]
    for k in [1, 2, 4, 5, 8, 10, 16, 25, 32, 50, 64]:
        for seed in range(5):
            indices = conll._sample_n_way_k_shots(
                dataset=conll._train, labels=labels, tag_type=tag_type, n=-1, k=k, seed=seed, return_indices=True
            )
            conll_indices[f"{k}-{seed}"] = indices

    with open("data/fewshot/fewshot_conll.json", "w") as f:
        json.dump(conll_indices, f)


def wnut():
    wnut_indices = {}
    wnut = WNUT_17()

    tag_type = "ner"
    label_dict = wnut.make_label_dictionary(tag_type, add_unk=False)
    labels = [label.decode("utf-8") for label in label_dict.idx2item]
    for k in [1, 2, 4, 5, 8, 10, 16, 25, 32, 50, 64]:
        for seed in range(5):
            indices = wnut._sample_n_way_k_shots(
                dataset=wnut._train, labels=labels, tag_type=tag_type, n=-1, k=k, seed=seed, return_indices=True
            )
            wnut_indices[f"{k}-{seed}"] = indices

    with open("data/fewshot/fewshot_wnut.json", "w") as f:
        json.dump(wnut_indices, f)


def ontonotes():
    ontonotes_indices = {}
    ontonotes = ONTONOTES()

    tag_type = "ner"
    label_dict = ontonotes.make_label_dictionary(tag_type, add_unk=False)
    labels = [label.decode("utf-8") for label in label_dict.idx2item]
    for k in [1, 2, 4, 5, 8, 10, 16, 25, 32, 50, 64]:
        for seed in range(5):
            indices = ontonotes._sample_n_way_k_shots(
                dataset=ontonotes._train, labels=labels, tag_type=tag_type, n=-1, k=k, seed=seed, return_indices=True
            )
            ontonotes_indices[f"{k}-{seed}"] = indices

    with open("data/fewshot/fewshot_ontonotes.json", "w") as f:
        json.dump(ontonotes_indices, f)


def fewnerd_coarse():
    fewnerd_indices = {}
    fewnerd = FEWNERD(
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
            "other-biologything": "other",
            "location-road/railway/highway/transit": "location",
            "person-actor": "person",
            "product-other": "product",
            "event-sportsevent": "event",
            "organization-government/governmentagency": "organization",
            "location-bodiesofwater": "location",
            "organization-media/newspaper": "organization",
            "art-music": "art",
            "other-chemicalthing": "other",
            "event-attack/battle/war/militaryconflict": "event",
            "organization-politicalparty": "organization",
            "art-writtenart": "art",
            "other-award": "other",
            "other-livingthing": "other",
            "event-other": "event",
            "art-film": "art",
            "product-software": "product",
            "organization-sportsleague": "organization",
            "other-language": "other",
            "other-disease": "other",
            "organization-showorganization": "organization",
            "product-airplane": "product",
            "other-astronomything": "other",
            "organization-religion": "organization",
            "product-car": "product",
            "person-scholar": "person",
            "other-currency": "other",
            "person-soldier": "person",
            "location-mountain": "location",
            "art-broadcastprogram": "art",
            "location-island": "location",
            "art-other": "art",
            "person-director": "person",
            "product-weapon": "product",
            "other-god": "other",
            "building-theater": "building",
            "other-law": "other",
            "product-food": "product",
            "other-medical": "other",
            "product-game": "product",
            "location-park": "location",
            "product-ship": "product",
            "building-sportsfacility": "building",
            "other-educationaldegree": "other",
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
    )

    tag_type = "ner"
    label_dict = fewnerd.make_label_dictionary(tag_type, add_unk=False)
    labels = [label.decode("utf-8") for label in label_dict.idx2item]
    for k in [1, 2, 4, 5, 8, 10, 16, 25, 32, 50, 64]:
        for seed in range(5):
            indices = fewnerd._sample_n_way_k_shots(
                dataset=fewnerd._train, labels=labels, tag_type=tag_type, n=-1, k=k, seed=seed, return_indices=True
            )
            fewnerd_indices[f"{k}-{seed}"] = indices

    with open("data/fewshot/fewshot_fewnerd_coarse.json", "w") as f:
        json.dump(fewnerd_indices, f)


def fewnerd_fine():
    fewnerd_indices = {}
    fewnerd = FEWNERD(
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
            "product-other": "other product",
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
    )

    tag_type = "ner"
    label_dict = fewnerd.make_label_dictionary(tag_type, add_unk=False)
    labels = [label.decode("utf-8") for label in label_dict.idx2item]
    for k in [1, 2, 4, 5, 8, 10, 16, 25, 32, 50, 64]:
        for seed in range(5):
            indices = fewnerd._sample_n_way_k_shots(
                dataset=fewnerd._train, labels=labels, tag_type=tag_type, n=-1, k=k, seed=seed, return_indices=True
            )
            fewnerd_indices[f"{k}-{seed}"] = indices

    with open("data/fewshot/fewshot_fewnerd_fine.json", "w") as f:
        json.dump(fewnerd_indices, f)


if __name__ == "__main__":
    main()
