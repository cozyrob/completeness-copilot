"""Feature engineering canonico della pipeline EBM v2.

La funzione OOF riceve i fold da ``split.py``: non crea suddivisioni proprie. Tutte le
feature che apprendono da TARGET sono elencate esplicitamente in ``TARGET_DERIVED``;
non si usa piu' il fragile criterio basato sul prefisso del nome.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import specie as S
from config import (
    IMPACT_M_GENERAL,
    IMPACT_M_OPERATION,
    K_STRUCT,
    TARGET_DERIVED,
)

M_ARTICOLO = 20
M_COMMA = 40
M_OPERA = 100

FERMA = set("""di a da in con su per tra fra il lo la i gli le un uno una del dello della
dei degli delle al allo alla ai agli alle dal dallo dalla dai dagli dalle nel nello nella
nei negli nelle sul sullo sulla sui sugli sulle e ed o ad che non si come piu anche tutti
tutto opere lavori intervento interventi realizzazione""".split())


def _norm(t):
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    return " ".join(w for w in re.findall(r"[a-z]{3,}", t) if w not in FERMA)


def carica(cartella) -> dict:
    cartella = Path(cartella)
    w = pd.read_parquet(cartella / "coppie_universo.parquet").drop(
        columns=["articolo"], errors="ignore"
    )
    pr = pd.read_parquet(cartella / "pratiche.parquet")
    U = pd.read_parquet(cartella / "universo.parquet")
    opere = pd.read_parquet(cartella / "opere.parquet")

    required_pr = {
        "Codice_Pratica", "Data_Presentazione", "Codice_Specie", "Descr_Specie",
        "Tipo_Intervento", "Oggetto",
    }
    required_w = {"Codice_Pratica", "documento", "al_deposito", "TARGET"}
    required_u = {"documento", "famiglia"}
    for frame, required, name in ((pr, required_pr, "pratiche"), (w, required_w, "coppie"),
                                  (U, required_u, "universo")):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Colonne mancanti in {name}: {sorted(missing)}")
    if pr.Codice_Pratica.duplicated().any():
        raise ValueError("Codice_Pratica non e' univoco in pratiche.parquet")
    if U.documento.duplicated().any():
        raise ValueError("documento non e' univoco in universo.parquet")

    DOC = sorted(U.documento.astype(str))
    idx = {d: i for i, d in enumerate(DOC)}
    P = pr.sort_values(
        ["Data_Presentazione", "Codice_Pratica"], kind="stable"
    ).reset_index(drop=True)
    P[["comma", "assenso", "paesaggistica", "cambio_uso"]] = pd.DataFrame(
        [S.decodifica(c, d) for c, d in zip(P.Codice_Specie, P.Descr_Specie)],
        index=P.index,
    )
    op = opere.sort_values(
        ["Codice_Pratica", "righe", "Cod_Opera"], ascending=[True, False, True]
    ).drop_duplicates("Codice_Pratica")
    P["opera"] = P.Codice_Pratica.map(
        op.set_index("Codice_Pratica").Cod_Opera
    ).fillna("ND")
    P["n_opere"] = P.Codice_Pratica.map(
        opere.groupby("Codice_Pratica").Cod_Opera.nunique()
    ).fillna(1)
    P["tipo_intervento"] = P.Tipo_Intervento.fillna("ND")
    P["testo"] = P.Oggetto.map(_norm)

    unknown_practices = set(w.Codice_Pratica) - set(P.Codice_Pratica)
    unknown_docs = set(w.documento.astype(str)) - set(DOC)
    if unknown_practices or unknown_docs:
        raise ValueError(
            f"Chiavi orfane: {len(unknown_practices)} pratiche, {len(unknown_docs)} documenti"
        )
    pi = {p: i for i, p in enumerate(P.Codice_Pratica)}
    DEP = np.zeros((len(P), len(DOC)), dtype=bool)
    TAR = np.zeros_like(DEP)
    for p, d, dep, target in zip(
        w.Codice_Pratica, w.documento.astype(str), w.al_deposito, w.TARGET
    ):
        i, j = pi[p], idx[d]
        DEP[i, j] |= bool(dep)
        TAR[i, j] |= bool(target)
    if np.any(DEP & TAR):
        raise ValueError("Una coppia pratica-documento e' sia depositata sia target")

    fam = U.assign(documento=U.documento.astype(str)).set_index("documento").famiglia
    fam = fam.reindex(DOC).astype(str).to_numpy()
    return {
        "P": P, "DEP": DEP, "TAR": TAR, "DOC": DOC, "nd": len(DOC),
        "npr": len(P), "fam": fam, "U": U.assign(documento=U.documento.astype(str)),
        "opere": opere.copy(),
    }


def _impact_base_rate(TAR, DEP, fit_mask):
    """P(target | candidato assente) specifica per documento."""
    eligible = np.asarray(fit_mask, dtype=bool)[:, None] & ~DEP
    counts = eligible.sum(0)
    successes = (TAR & eligible).sum(0)
    return np.divide(
        successes,
        counts,
        out=np.zeros(TAR.shape[1], dtype=float),
        where=counts > 0,
    )


def _impact_smoothed_rate(TAR, DEP, fit_mask, group_mask, parent, m):
    selected = np.asarray(fit_mask, dtype=bool) & np.asarray(group_mask, dtype=bool)
    eligible = selected[:, None] & ~DEP
    counts = eligible.sum(0)
    successes = (TAR & eligible).sum(0)
    return (successes + m * parent) / (counts + m)


def _impact_single_context(TAR, DEP, fit_mask, keys, m, zero_levels=()):
    keys = np.asarray(keys, dtype=str)
    parent = _impact_base_rate(TAR, DEP, fit_mask)
    result = np.zeros(TAR.shape, dtype=float)
    zero_levels = {str(value) for value in zero_levels}
    for value in np.unique(keys):
        if value in zero_levels:
            continue
        group = keys == value
        estimate = _impact_smoothed_rate(
            TAR, DEP, fit_mask, group, parent, m
        )
        result[group] = estimate - parent
    return result


def _impact_regime_intervento(TAR, DEP, fit_mask, article, intervention):
    """Impact gerarchico documento -> articolo -> tipo di intervento."""
    article = np.asarray(article, dtype=str)
    intervention = np.asarray(intervention, dtype=str)
    parent = _impact_base_rate(TAR, DEP, fit_mask)
    article_rate = {}
    for value in np.unique(article):
        article_rate[value] = _impact_smoothed_rate(
            TAR,
            DEP,
            fit_mask,
            article == value,
            parent,
            IMPACT_M_GENERAL,
        )
    result = np.zeros(TAR.shape, dtype=float)
    joint = np.char.add(np.char.add(article, "||"), intervention)
    for value in np.unique(joint):
        group = joint == value
        row = int(np.flatnonzero(group)[0])
        article_value = article[row]
        if intervention[row] == "ND":
            estimate = article_rate[article_value]
        else:
            estimate = _impact_smoothed_rate(
                TAR,
                DEP,
                fit_mask,
                group,
                article_rate[article_value],
                IMPACT_M_GENERAL,
            )
        result[group] = estimate - parent
    return result


def _operation_memberships(D):
    operations = D.get("opere")
    if operations is None:
        raise ValueError("Dati delle opere mancanti: impossibile costruire l'impact")
    practice_index = {
        value: i for i, value in enumerate(D["P"].Codice_Pratica)
    }
    memberships = {}
    for operation, practices in operations.groupby("Cod_Opera").Codice_Pratica:
        mask = np.zeros(D["npr"], dtype=bool)
        for practice in practices.unique():
            if practice in practice_index:
                mask[practice_index[practice]] = True
        memberships[str(operation)] = mask
    return memberships


def _impact_opere(D, fit_mask):
    TAR, DEP = D["TAR"], D["DEP"]
    parent = _impact_base_rate(TAR, DEP, fit_mask)
    result = np.full(TAR.shape, -np.inf, dtype=float)
    has_operation = np.zeros(D["npr"], dtype=bool)
    memberships = _operation_memberships(D)
    for operation in sorted(memberships):
        group = memberships[operation]
        has_operation |= group
        support = int((np.asarray(fit_mask, dtype=bool) & group).sum())
        impact = (
            _impact_smoothed_rate(
                TAR,
                DEP,
                fit_mask,
                group,
                parent,
                IMPACT_M_OPERATION,
            )
            - parent
            if support
            else np.zeros(D["nd"], dtype=float)
        )
        result[group] = np.maximum(result[group], impact)
    result[~has_operation] = 0.0
    if np.isneginf(result).any():
        raise AssertionError("Impact opere non assegnato")
    return result


def _regime_autorizzativo(P):
    result = np.full(len(P), "non_dichiarato", dtype=object)
    landscape = P.paesaggistica.astype(bool).to_numpy()
    consent = P.assenso.astype(str).to_numpy()
    result[consent == "richiesta"] = "assenso_richiesto"
    result[consent == "non_necessari"] = "assenso_non_necessario"
    result[landscape] = "paesaggistica"
    return result.astype(str)


def _prior_gerarchico(TAR, tr, chiavi, attenuazioni):
    if not np.any(tr):
        raise ValueError("Il fit mask non contiene pratiche")
    padre = np.tile(TAR[tr].mean(0), (len(TAR), 1))
    for key, m in zip(chiavi, attenuazioni):
        nuovo = padre.copy()
        for gruppo in np.unique(key):
            msk = tr & (key == gruppo)
            n = int(msk.sum())
            if n:
                riga = np.flatnonzero(key == gruppo)[0]
                nuovo[key == gruppo] = (n * TAR[msk].mean(0) + m * padre[riga]) / (n + m)
        padre = nuovo
    return padre


def _vicini(sim, target_train, idx_sel, idx_train, k, fallback):
    if sim.shape[1] == 0:
        raise ValueError("Nessun vicino disponibile")
    sim = np.where(idx_sel[:, None] == idx_train[None, :], -1.0, sim)
    k = min(k, sim.shape[1])
    ordine = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    out = np.zeros((sim.shape[0], target_train.shape[1]), dtype=float)
    for i, neighbors in enumerate(ordine):
        weights = np.maximum(sim[i, neighbors], 0)
        out[i] = (weights @ target_train[neighbors]) / weights.sum() \
            if weights.sum() > 0 else fallback
    return out


def costruisci(D, fit_mask):
    P, DEP, TAR, fam = D["P"], D["DEP"], D["TAR"], np.asarray(D["fam"])
    npr, nd = DEP.shape
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if len(fit_mask) != npr or not fit_mask.any():
        raise ValueError("fit_mask non valido")
    Df, Tf = DEP.astype(float), TAR.astype(float)
    comma = P.comma.astype(str).to_numpy()
    art = np.where(comma == "23", "art.23", "art.22")

    def stendi(v):
        return np.repeat(np.asarray(v, dtype=float).reshape(-1, 1), nd, axis=1)

    col = {}
    fam_codes = {v: i for i, v in enumerate(sorted(set(fam)))}
    col["doc_famiglia"] = np.tile(
        np.array([fam_codes[v] for v in fam], dtype=float), (npr, 1)
    )
    for name, series in (
        ("pr_comma", P.comma), ("pr_opera", P.opera),
        ("pr_tipo_intervento", P.tipo_intervento), ("pr_assenso", P.assenso),
        ("pr_cambio_uso", P.cambio_uso),
    ):
        values = series.astype(str)
        codes = {v: i for i, v in enumerate(sorted(values.unique()))}
        col[name] = stendi(values.map(codes).to_numpy())
    col["pr_paesaggistica"] = stendi(P.paesaggistica.astype(int).to_numpy())
    col["pr_n_opere"] = stendi(P.n_opere.to_numpy())
    col["pr_n_documenti_deposito"] = stendi(DEP.sum(1))

    col["impact_doc_regime_intervento"] = _impact_regime_intervento(
        TAR,
        DEP,
        fit_mask,
        art,
        P.tipo_intervento.astype(str).to_numpy(),
    )
    col["impact_doc_opere"] = _impact_opere(D, fit_mask)
    col["impact_doc_regime_autorizzativo"] = _impact_single_context(
        TAR,
        DEP,
        fit_mask,
        _regime_autorizzativo(P),
        IMPACT_M_GENERAL,
    )
    col["impact_doc_cambio_uso"] = _impact_single_context(
        TAR,
        DEP,
        fit_mask,
        P.cambio_uso.astype(str).to_numpy(),
        IMPACT_M_GENERAL,
        zero_levels=("non_dichiarato",),
    )

    family_deposit = np.zeros((npr, nd), dtype=float)
    for value in np.unique(fam):
        same_family = fam == value
        family_deposit[:, same_family] = DEP[:, same_family].sum(1, keepdims=True)
    col["int_famiglia_al_deposito"] = family_deposit

    col["doc_freq_mancante"] = np.tile(Tf[fit_mask].mean(0), (npr, 1))
    col["doc_freq_deposito"] = np.tile(Df[fit_mask].mean(0), (npr, 1))
    comma_opera = np.array([
        f"{c}|{o}" for c, o in zip(comma, P.opera.astype(str).to_numpy())
    ])
    col["int_prior_comma_opera"] = _prior_gerarchico(
        TAR, fit_mask, [art, comma, comma_opera], [M_ARTICOLO, M_COMMA, M_OPERA]
    )

    absent = (~DEP).astype(float)
    cooc_absent = np.zeros((npr, nd), dtype=float)
    knn = np.zeros((npr, nd), dtype=float)
    for article in ("art.22", "art.23"):
        train_article = fit_mask & (art == article)
        selected = art == article
        if not train_article.any():
            raise ValueError(f"Nessuna pratica di training per {article}")
        A_train, T_train = absent[train_article], Tf[train_article]
        matrix = (A_train.T @ T_train) / np.maximum(A_train.sum(0)[:, None], 1)
        cooc_absent[selected] = (
            absent[selected] @ matrix
        ) / np.maximum(absent[selected].sum(1, keepdims=True), 1)

        D_train = Df[train_article]
        intersection = Df[selected] @ D_train.T
        union = Df[selected].sum(1, keepdims=True) + D_train.sum(1)[None, :] - intersection
        jaccard = intersection / np.maximum(union, 1)
        knn[selected] = _vicini(
            jaccard,
            T_train,
            np.flatnonzero(selected),
            np.flatnonzero(train_article),
            K_STRUCT,
            T_train.mean(0),
        )
    col["int_cooccorrenza_assenti"] = cooc_absent
    col["int_knn"] = knn
    return col


def costruisci_out_of_fold(D, train_mask, fold_ids):
    """Costruisce tutte le feature, sostituendo nel train le supervisionate OOF."""
    train_mask = np.asarray(train_mask, dtype=bool)
    fold_ids = np.asarray(fold_ids)
    if len(train_mask) != D["npr"] or len(fold_ids) != D["npr"]:
        raise ValueError("Maschere OOF con dimensione errata")
    if np.any(fold_ids[~train_mask] != -1):
        raise ValueError("Le pratiche fuori training devono avere fold -1")
    col = costruisci(D, train_mask)
    for fold in sorted(np.unique(fold_ids[train_mask])):
        held = train_mask & (fold_ids == fold)
        sub_train = train_mask & ~held
        folded = costruisci(D, sub_train)
        for name in TARGET_DERIVED:
            # La feature semantica e' supervisionata ma appartiene a testo.py, che
            # riceve gli stessi fold ed esegue separatamente questo identico passaggio.
            if name in col:
                col[name][held] = folded[name][held]
    return col
