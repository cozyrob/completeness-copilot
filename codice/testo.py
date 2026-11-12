"""Feature semantica TF-IDF con lo stesso protocollo OOF delle altre feature."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import feature2 as F2
from config import K_SEMANTIC

FERMA_PIU = set("""dell all nell sull dall coll quell nella dello della dei degli delle
allo alla agli alle nello nelle sullo sulla sugli sulle dallo dalla dagli dalle quello
quella cui suo sua loro tale tali stesso stessa presente presenti essere sono stato stata
stati state viene vengono verra verranno oltre inoltre quindi pertanto seguito mediante
tramite ovvero nonche altresi due tre quattro cinque sei sette otto nove dieci""".split())
FERMA = F2.FERMA | FERMA_PIU


def ripara(testo):
    testo = str(testo).replace("\xa0", " ")
    testo = re.sub(r"(?<![A-Za-zÀ-ÿ])¿(?![A-Za-zÀ-ÿ])", " E ", testo)
    testo = testo.replace("¿", "'")
    return re.sub(r"\s+", " ", testo).strip()


def norma_pulita(testo):
    testo = unicodedata.normalize("NFKD", ripara(testo))
    testo = testo.encode("ascii", "ignore").decode().lower()
    return " ".join(w for w in re.findall(r"[a-z]{3,}", testo) if w not in FERMA)


def vett_caratteri():
    return TfidfVectorizer(
        min_df=3, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
    )


def similarita_tfidf(testi, fit_mask):
    """Adatta vocabolario e IDF esclusivamente alle pratiche indicate da fit_mask."""
    fit_mask = np.asarray(fit_mask, dtype=bool)
    vec = vett_caratteri()
    train = vec.fit_transform(np.asarray(testi, dtype=object)[fit_mask])
    all_rows = vec.transform(np.asarray(testi, dtype=object))
    return (all_rows @ train.T).toarray()


def colonna(D, fit_mask, testi):
    fit_mask = np.asarray(fit_mask, dtype=bool)
    target = D["TAR"].astype(float)
    return F2._vicini(
        similarita_tfidf(testi, fit_mask),
        target[fit_mask],
        np.arange(D["npr"]),
        np.flatnonzero(fit_mask),
        K_SEMANTIC,
        target[fit_mask].mean(0),
    )


def colonna_out_of_fold(D, train_mask, testi, fold_ids):
    """Ricalcola il TF-IDF e il voto target in ciascun fold fornito da split.py."""
    train_mask = np.asarray(train_mask, dtype=bool)
    fold_ids = np.asarray(fold_ids)
    if np.any(fold_ids[~train_mask] != -1):
        raise ValueError("Le pratiche fuori training devono avere fold -1")
    semantic = colonna(D, train_mask, testi)
    for fold in sorted(np.unique(fold_ids[train_mask])):
        held = train_mask & (fold_ids == fold)
        sub_train = train_mask & ~held
        folded = colonna(D, sub_train, testi)
        semantic[held] = folded[held]
    return semantic
