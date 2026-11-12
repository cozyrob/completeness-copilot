"""Definizione dichiarata e riproducibile dell'universo documentale a 149 classi.

Le esclusioni correggono la verita' di riferimento: non sono potatura guidata dalle
prestazioni. L'accorpamento unisce due codici che rappresentano lo stesso elaborato.
"""
import numpy as np

FUORI_FAMIGLIE = {"SIS"}
FUORI_CODICI = {"ELT001", "ELG041", "ELG073", "ELG020"}
ACCORPA = {"ELG012": "ELG001"}


def escludi(D, codici=(), famiglie=(), accorpa=None):
    DOC = np.asarray(D["DOC"])
    fam = np.asarray(D["fam"])
    TAR, DEP = D["TAR"].copy(), D["DEP"].copy()
    assorbiti = []
    for assorbito, principale in (accorpa or {}).items():
        if assorbito not in DOC or principale not in DOC:
            continue
        ja, jb = list(DOC).index(assorbito), list(DOC).index(principale)
        DEP[:, jb] = DEP[:, jb] | DEP[:, ja]
        TAR[:, jb] = (TAR[:, jb] | TAR[:, ja]) & ~DEP[:, jb]
        assorbiti.append(assorbito)
    fuori = np.isin(DOC, list(codici) + assorbiti) | np.isin(fam, list(famiglie))
    tieni = ~fuori
    N = dict(D)
    N["DOC"] = list(DOC[tieni])
    N["fam"] = fam[tieni]
    N["TAR"] = TAR[:, tieni]
    N["DEP"] = DEP[:, tieni]
    N["nd"] = int(tieni.sum())
    N["U"] = D["U"][D["U"].documento.isin(N["DOC"])].copy()
    N["_esclusi"] = list(DOC[fuori])
    return N


def definitivo(D):
    return escludi(D, FUORI_CODICI, FUORI_FAMIGLIE, ACCORPA)
