# CC_finale — CompletenessCopilot, versione finale

Copia pulita e minima della pipeline **EBM additivo a 13 variabili**
(origine: `../CompletenessCopilot_EBM_v2/`), con il codice separato dai file
testuali. Le cartelle originali non sono state modificate.

Il risultato finale è la **valutazione sul primo semestre 2026**: modello riaddestrato
su tutte le pratiche precedenti al 1° gennaio 2026 (3.314) e valutato sulle pratiche
presentate dal 1° gennaio al 26 giugno 2026 (740, di cui 251 incomplete).
Riferimento: R@3 = 58,49% (run `out/h1_2026_finale`).

## Struttura

```
CC_finale/
├── README.md                  questo file
├── codice/                    tutto ciò che serve per far girare il modello
│   ├── config.py              parametri, 13 feature, protocollo temporale
│   ├── run_ebm.py             entrypoint (check / evaluate)
│   ├── verify_run.py          verifica read-only di un run pubblicato
│   ├── feature2.py            feature engineering out-of-fold
│   ├── testo.py               feature semantica TF-IDF
│   ├── specie.py              decodifica specie edilizia
│   ├── universo.py            universo documentale a 149 classi
│   ├── split.py               split temporali e fold interni
│   ├── valuta.py              metriche di ranking e bootstrap
│   ├── provenienza.py         hash e versioni dei pacchetti
│   ├── requirements.lock      versioni esatte delle librerie
│   ├── tests/test_pipeline.py 7 test automatici
│   ├── dati/                  estrazione art. 22/23 fino ad agosto 2026 (4 parquet + manifest)
│   └── out/h1_2026_finale/    run finale di riferimento
├── documentazione/
│   ├── README_pipeline.md                     scelte metodologiche (scritto per lo snapshot 2025)
│   ├── IMPLEMENTAZIONE_MODELLO_DEFINITIVO_13.md
│   └── FILE_NON_USATI.md                      inventario di ciò che è stato escluso
└── archivio/
    ├── snapshot_2025/         vecchio snapshot (con istruttore_cap.parquet) e run di validazione Q2 2025
    └── manifest_estrazione_2026_originale.json
```

`dati/` e `out/` restano dentro `codice/` perché `config.py` li cerca accanto ai sorgenti.

## Dati (non inclusi nel repository)

Il repository contiene solo codice e documentazione. I dati delle pratiche edilizie, i
risultati dei run (`out/`), i modelli addestrati, `archivio/` e
`documentazione/FILE_NON_USATI.md` restano in locale: li esclude `.gitignore`.

Per far girare la pipeline servono, in `codice/dati/`:

| File | Colonne richieste |
|---|---|
| `pratiche.parquet` | `Codice_Pratica`, `Data_Presentazione`, `Codice_Specie`, `Descr_Specie`, `Tipo_Intervento`, `Oggetto` |
| `coppie_universo.parquet` | `Codice_Pratica`, `documento`, `al_deposito`, `TARGET` |
| `universo.parquet` | `documento`, `famiglia` |
| `opere.parquet` | `Codice_Pratica`, `Cod_Opera`, `righe` |
| `manifest.json` | `{"sha256": {"<file>.parquet": "<hash SHA-256>", ...}}` per ogni parquet |

Anche i test in `tests/` leggono i dati reali, quindi senza `dati/` non partono.

## Differenze rispetto a `CompletenessCopilot_EBM_v2`

- **Dati**: si usa la nuova estrazione (`analisi/valutazione_2026/dati_h1_2026`, ricavata da
  `Dataset_nuove variabili/Dati_ts_con_var_aggiunte_22_23.xlsx`) invece dello snapshot
  2025. `istruttore_cap.parquet`, mai letto dal modello, è stato archiviato.
- **Fase `h1_2026`** aggiunta a `run_ebm.py`, `split.py` e `config.py`: rifit su
  `Data_Presentazione < 2026-01-01`, valutazione su `2026-01-01 ≤ data ≤ 2026-06-26`.
  Le pratiche successive (49) hanno meno di 90 giorni di osservazione e sono escluse.
- I conteggi attesi in `tests/test_pipeline.py` sono aggiornati alla nuova estrazione.

Le fasi `validation` (Q2 2025) e `test` (H2 2025) funzionano ancora, ma sulla nuova
estrazione danno numeri leggermente diversi da quelli calcolati sullo snapshot 2025
(es. R@3 di validazione 53,64% nel run archiviato).

## Ambiente

```bash
python3 -m pip install -r codice/requirements.lock
```

Verificato con Python 3.14 e le versioni elencate in `requirements.lock`.

## Uso (da dentro `codice/`)

```bash
python3 run_ebm.py check
python3 run_ebm.py evaluate --stage h1_2026 --run-id <nuovo_id> --acknowledge-retrospective-test
python3 verify_run.py out/h1_2026_finale
python3 -m unittest discover -s tests -v
```

Il run finale richiede circa 20 minuti. Ogni run viene scritto in `out/<run-id>/` e non
sovrascrive mai un run esistente. La coorte H1 2026 è già stata consultata: non va usata
per scegliere feature, iperparametri o soglie continuando a chiamarla valutazione cieca.
