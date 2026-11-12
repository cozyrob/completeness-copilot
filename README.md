# Completeness Copilot

Sistema di ranking dei documenti mancanti nelle pratiche edilizie (SCIA art. 22 e 23).
Per ogni pratica depositata, il modello ordina i 149 tipi di documento che potrebbero
essere richiesti in integrazione, così da segnalare in cima alla lista quelli più
probabilmente mancanti.

Il modello è un **Explainable Boosting Machine** (EBM) additivo a 13 variabili
(`interactions=0`), scelto perché ogni contributo resta leggibile.

## Risultato finale

Modello riaddestrato su tutte le pratiche presentate prima del 1° gennaio 2026 (3.314) e
valutato sulle pratiche presentate dal 1° gennaio al 26 giugno 2026 (740, di cui 251
incomplete). Le metriche di recall, hit rate, MRR e R-precision sono calcolate sulle
pratiche incomplete. Intervalli bootstrap al 95% su 2.000 ricampionamenti di pratiche
intere.

| Metrica | Valore | IC 95% |
|---|---:|---:|
| Recall@3 | 58,49% | 53,17–63,79 |
| Hit@3 | 73,71% | 68,35–79,09 |
| Recall@5 | 66,22% | 61,29–71,21 |
| Hit@5 | 80,08% | 75,00–85,14 |
| Recall@10 | 73,80% | 68,95–77,97 |
| Hit@10 | 85,66% | 81,12–89,78 |
| MRR | 72,03% | 67,05–76,77 |
| R-precision | 54,39% | 49,15–59,50 |

Il primo semestre 2026 è già stato consultato durante lo sviluppo: è un benchmark
temporale retrospettivo, non una valutazione cieca.

## Struttura del repository

```
├── codice/
│   ├── config.py              parametri, 13 feature, protocollo temporale
│   ├── run_ebm.py             entrypoint (check / evaluate)
│   ├── verify_run.py          verifica read-only di un run
│   ├── feature2.py            feature engineering out-of-fold
│   ├── testo.py               feature semantica TF-IDF sull'oggetto della pratica
│   ├── specie.py              decodifica della specie edilizia
│   ├── universo.py            universo documentale a 149 classi
│   ├── split.py               split temporali e fold interni
│   ├── valuta.py              metriche di ranking e bootstrap
│   ├── provenienza.py         hash e versioni dei pacchetti
│   ├── requirements.lock      versioni esatte delle librerie
│   └── tests/test_pipeline.py test automatici
└── documentazione/
    ├── README_pipeline.md                     scelte metodologiche e protocollo
    └── IMPLEMENTAZIONE_MODELLO_DEFINITIVO_13.md  definizione delle 13 variabili
```

## Dati (non inclusi)

Il repository contiene solo codice e documentazione: i dati delle pratiche, i risultati
dei run e i modelli addestrati non sono pubblicati. Per far girare la pipeline vanno
messi in `codice/dati/`:

| File | Colonne richieste |
|---|---|
| `pratiche.parquet` | `Codice_Pratica`, `Data_Presentazione`, `Codice_Specie`, `Descr_Specie`, `Tipo_Intervento`, `Oggetto` |
| `coppie_universo.parquet` | `Codice_Pratica`, `documento`, `al_deposito`, `TARGET` |
| `universo.parquet` | `documento`, `famiglia` |
| `opere.parquet` | `Codice_Pratica`, `Cod_Opera`, `righe` |
| `manifest.json` | `{"sha256": {"<file>.parquet": "<hash SHA-256>", ...}}` per ogni parquet |

All'avvio la pipeline confronta gli hash dei parquet con `manifest.json` e si ferma se
non coincidono. Anche i test leggono i dati reali, quindi senza `codice/dati/` non partono.

## Installazione

```bash
python3 -m pip install -r codice/requirements.lock
```

Verificato con Python 3.14.

## Uso (da dentro `codice/`)

```bash
python3 run_ebm.py check
python3 run_ebm.py evaluate --stage h1_2026 --run-id <id> --acknowledge-retrospective-test
python3 verify_run.py out/<id>
python3 -m unittest discover -s tests -v
```

Fasi disponibili per `--stage`:

| Fase | Addestramento | Valutazione |
|---|---|---|
| `validation` | prima del 1/4/2025 | 1/4/2025 – 30/6/2025 |
| `test` | prima del 1/7/2025 | 1/7/2025 – 31/12/2025 |
| `h1_2026` (finale) | prima del 1/1/2026 | 1/1/2026 – 26/6/2026 |

Le pratiche successive al 26 giugno 2026 hanno meno di 90 giorni di osservazione e sono
escluse. Il run finale richiede circa 20 minuti. Ogni run viene scritto in `out/<id>/`
con punteggi, modello, metriche, top 10 e manifest con gli hash di dati e codice, e non
sovrascrive mai un run esistente.
