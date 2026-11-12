"""Decodifica della specie edilizia nelle variabili dichiarative usate dal modello."""
import re

BASI = {304: "22c1", 330: "23"}
RAMI = ["non_necessari", "richiesta", "paesaggistica"]
CAMBI = ["nessuno", "ui_stessa_categoria", "ui_diversa_categoria", "immobile_stessa_categoria"]


def _comma_da_testo(testo):
    if "art.23" in testo or "art. 23" in testo:
        return "23"
    if "22 c.2 bis" in testo or "22 comma 2 bis" in testo:
        return "22c2bis"
    if "22 c.2" in testo or "22 comma 2" in testo:
        return "22c2"
    if "22 c.1" in testo or "22 comma 1" in testo:
        return "22c1"
    return "altro"


def decodifica(codice, descrizione):
    try:
        codice = int(codice)
    except (TypeError, ValueError):
        codice = -1
    for base, comma in BASI.items():
        if base <= codice <= base + 11:
            offset = codice - base
            ramo = RAMI[offset // 4]
            return (
                comma,
                "non_dichiarato" if ramo == "paesaggistica" else ramo,
                ramo == "paesaggistica",
                CAMBI[offset % 4],
            )
    testo = re.sub(r"\s+", " ", str(descrizione)).lower()
    comma = _comma_da_testo(testo)
    if "paesaggistic" in testo:
        return comma, "non_dichiarato", True, "non_dichiarato"
    if "atti di assenso" in testo and "senza atti di assenso" not in testo:
        return comma, "richiesta", False, "non_dichiarato"
    return comma, "non_necessari", False, "non_dichiarato"
