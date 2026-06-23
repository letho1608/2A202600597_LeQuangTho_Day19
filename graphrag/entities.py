import re
from typing import List, Tuple, Dict

ENTITIES = {
    "COMPANY": [
        "Tesla", "Ford", "General Motors", "BMW", "Hyundai", "Kia",
        "Mercedes-Benz", "Rivian", "Vinfast", "Cadillac", "Lexus", "Audi",
        "Chevrolet", "Toyota", "Honda", "Nissan", "Volkswagen", "Porsche",
        "Lucid", "BYD", "Stellantis", "Subaru", "Mazda", "Volvo",
        "Polestar", "Jaguar", "Cox Automotive"
    ],
    "ORGANIZATION": [
        "ICCT", "EPA", "DOE", "NREL", "BloombergNEF", "McKinsey",
        "Kelley Blue Book", "Alternative Fuels Data Center", "NHTSA", "CARB",
        "International Council on Clean Transportation", "National Renewable Energy Laboratory"
    ],
    "PERSON": [
        "Elon Musk", "Sam Altman", "Stephanie Valdez Streaty", "Anh Bui",
        "Peter Slowik", "Nic Lutsey"
    ],
    "LOCATION": [
        "United States", "California", "China", "Europe", "Germany", "Norway",
        "Netherlands", "France", "UK", "Japan", "South Korea", "India",
        "Texas", "New York", "Florida"
    ],
    "TECHNOLOGY": [
        "BEV", "Battery Electric Vehicle", "PHEV", "Plug-in Hybrid Electric Vehicle",
        "HEV", "Hybrid Electric Vehicle", "ICE", "Internal Combustion Engine",
        "EVSE", "Charging Station", "Fast Charging", "Lithium-ion",
        "Regenerative Braking", "FCEV", "Solid-state battery"
    ],
    "METRIC": ["MPGe", "kWh", "GWh", "CO2"],
}

SYNONYMS = {
    "U.S.": "United States", "USA": "United States", "GM": "General Motors",
    "Chevy": "Chevrolet", "Mercedes": "Mercedes-Benz",
    "BEV": "Battery Electric Vehicle", "PHEV": "Plug-in Hybrid Electric Vehicle",
    "HEV": "Hybrid Electric Vehicle", "ICE": "Internal Combustion Engine",
}

ALL_ENTS: Dict[str, str] = {}
for cat, items in ENTITIES.items():
    for item in items:
        c = SYNONYMS.get(item, item)
        ALL_ENTS[c] = cat
        if c != item:
            ALL_ENTS[item] = cat


def extract_entities(text: str) -> List[Tuple[str, str]]:
    found = []
    for entity, cat in sorted(ALL_ENTS.items(), key=lambda x: -len(x[0])):
        if re.search(re.escape(entity), text, re.IGNORECASE):
            found.append((SYNONYMS.get(entity, entity), cat))
    return list(set(found))


def extract_relations(text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    rels = []
    names = [e[0] for e in entities]
    for sent in re.split(r'[.!?]+', text):
        sent_ents = list(set(e for e in names if re.search(re.escape(e), sent, re.IGNORECASE)))
        for i in range(len(sent_ents)):
            for j in range(i + 1, len(sent_ents)):
                rels.append((sent_ents[i], "RELATED_TO", sent_ents[j]))
    for year in re.findall(r'\b20\d{2}\b', text):
        for n in names:
            skip = {"BEV", "PHEV", "HEV", "EVSE", "MPGe", "kWh", "GWh", "CO2", "ICE"}
            if n not in skip:
                ecat = next((c for c, items in ENTITIES.items() if n in items), None)
                if ecat in ("COMPANY", "ORGANIZATION"):
                    rels.append((n, "FOUNDED_IN", year))
    for n in names:
        ecat = next((c for c, items in ENTITIES.items() if n in items), None)
        if ecat in ("COMPANY", "ORGANIZATION"):
            for loc in names:
                if loc in ENTITIES["LOCATION"] and n != loc:
                    rels.append((n, "LOCATED_IN", loc))
    return rels
