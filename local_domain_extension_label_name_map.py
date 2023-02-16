from flair.data import Corpus
from flair.datasets import CONLL_03, FEWNERD, WNUT_17, ColumnCorpus


def get_corpus(name: str, map: str = "short", path: str = "/") -> Corpus:
    if name == "conll03":

        if map == "short":
            return CONLL_03(
                base_path=f"{path}/datasets",
                in_memory=True,
                label_name_map={
                    "LOC": "location",
                    "ORG": "organization",
                    "PER": "person",
                    "MISC": "miscellaneous",
                },
            )
        if map == "long":
            return CONLL_03(
                base_path=f"{path}/datasets",
                in_memory=True,
                label_name_map={
                    "LOC": "location name",
                    "ORG": "organization name",
                    "PER": "person name",
                    "MISC": "other name (not person name, not organization name, not location name)",
                },
            )

    if name == "ontonotes":

        if map == "short":
            return ColumnCorpus(
                f"{path}/datasets/onto-ner",
                column_format={0: "text", 1: "pos", 2: "upos", 3: "ner"},
                label_name_map={
                    "CARDINAL": "cardinal",
                    "DATE": "date",
                    "EVENT": "event",
                    "FAC": "facility",
                    "GPE": "geo-political entity",
                    "LANGUAGE": "language",
                    "LAW": "law",
                    "LOC": "location",
                    "MONEY": "money",
                    "NORP": "affiliation",
                    "ORDINAL": "ordinal",
                    "ORG": "organization",
                    "PERCENT": "percent",
                    "PERSON": "person",
                    "PRODUCT": "product",
                    "QUANTITY": "quantity",
                    "TIME": "time",
                    "WORK_OF_ART": "work of Art",
                },
            )

        if map == "long":
            return ColumnCorpus(
                f"{path}/datasets/onto-ner",
                column_format={0: "text", 1: "pos", 2: "upos", 3: "ner"},
                label_name_map={
                    "CARDINAL": "cardinal value",
                    "DATE": "reference to a date or period",
                    "EVENT": "event name",
                    "FAC": "name of man-made structure or facility",
                    "GPE": "name of country, city, state, province or municipality",
                    "LANGUAGE": "language",
                    "LAW": "named treaty or chapter of named legal document",
                    "LOC": "name of geographical location",
                    "MONEY": "monetary value",
                    "NORP": "adjectival form of named religion, heritage, geographical or political affiliation",
                    "ORDINAL": "ordinal number or adverbial",
                    "ORG": "organization name",
                    "PERCENT": "percent value",
                    "PERSON": "person name",
                    "PRODUCT": "product name",
                    "QUANTITY": "quantity value",
                    "TIME": "time reference",
                    "WORK_OF_ART": "title of book, song, movie or award",
                },
            )

    if name == "wnut17":
        if map == "short":
            return WNUT_17(
                label_name_map={
                    "location": "Location",
                    "corporation": "Corporation",
                    "person": "Person",
                    "creative-work": "Creative Work",
                    "product": "Product",
                    "group": "Group",
                }
            )

        if map == "long":
            return WNUT_17(
                label_name_map={
                    "location": "location name",
                    "corporation": "corporation name",
                    "person": "person name",
                    "creative-work": "name of song, movie, book or other creative work",
                    "product": "name of product or consumer good",
                    "group": "name of music band, sports team or non-corporate organization",
                }
            )

    if name == "fewnerd":
        return FEWNERD(
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
                "art-broadcastprogramm": "art broadcastprogramm",
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
        )


def get_label_name_map(corpus: str):
    if corpus == "conll03":
        label_name_map = {
            "LOC": "location",
            "ORG": "organization",
            "PER": "person",
            "MISC": "miscellaneous",
        }
    elif corpus == "wnut17":
        label_name_map = {
            "location": "Location",
            "corporation": "Corporation",
            "person": "Person",
            "creative-work": "Creative Work",
            "product": "Product",
            "group": "Group",
        }
    else:
        raise Exception("unknown corpus")
    return label_name_map
