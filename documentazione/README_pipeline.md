# CompletenessCopilot EBM v2

Implementazione canonica e isolata del sistema di ranking documentale. La cartella
originale non viene modificata.

## Decisioni metodologiche incorporate

- Il modello resta un classificatore binario EBM usato per ordinare i documenti. Questa
  scelta e' coerente con l'obiettivo di ispezionabilita' e, nel confronto disponibile a
  parita' di protocollo, l'EBM ha superato XGBoost pairwise. Non si afferma che il
  pointwise sia in assoluto superiore a ogni ranker.
- Le feature definitive sono 13 e l'EBM resta additivo (`interactions=0`).
  `doc_identita` e' inclusa; `int_prior_istruttore` e le otto variabili `pr_*` sono
  escluse dal modello. Il file sorgente con l'istruttore resta nello snapshot solo per
  provenienza e non viene letto dal codice.
- Quattro regularized impact encoding rendono il contesto specifico del documento
  candidato: regime di intervento, singole opere, regime autorizzativo e cambio d'uso.
  Ogni valore e' la differenza tra una m-estimate contestuale e la probabilita' generale
  del documento fra le sole pratiche in cui era assente al deposito. Si usa `m=40` per
  i contesti ordinari e `m=100` per le opere; nelle pratiche con piu' opere si conserva
  il massimo impact. Il cambio d'uso non dichiarato vale zero.
- La pratica (`Codice_Pratica`) e' l'unita' di campionamento. La particella catastale non
  viene usata per raggruppare ne' gli split principali ne' i fold interni.
- Un solo vettore di fold interni, stratificato per pratica completa/incompleta, governa
  tutte le feature supervisionate e il TF-IDF.
- `doc_freq_mancante` e i quattro impact sono supervisionati e vengono calcolati
  out-of-fold. Le feature supervisionate sono dichiarate esplicitamente in
  `config.TARGET_DERIVED`.
- `int_cooccorrenza_assenti` mantiene la semantica dichiarata: aggrega, sui documenti
  assenti nella pratica, la stima `P(documento B mancante | documento A assente)` appresa
  solo sul training pertinente.
- I documenti gia' presenti al deposito sono sempre esclusi dai candidati, ma la loro
  identita' e il loro insieme restano disponibili per costruire le feature.
- `pr_paesaggistica` e' trattata come nominale binaria.

## Protocollo di valutazione

Il percorso principale e' temporale:

1. sviluppo: date precedenti al 1 aprile 2025;
2. validazione: 1 aprile–30 giugno 2025;
3. refit: date precedenti al 1 luglio 2025;
4. test retrospettivo: 1 luglio–31 dicembre 2025;
5. pratiche dal 1 gennaio 2026 escluse dal protocollo corrente.

Il semestre H2 2025 e' gia' stato consultato negli esperimenti storici, quindi non e' un
blind test. Nel testo della tesi va chiamato *benchmark temporale retrospettivo*. Una
conferma finale realmente indipendente richiede una futura estrazione matura mai usata
per scelte di modello. La CV ripetuta 5x5 non fa parte della pipeline canonica; una sola
CV a 5 fold puo' essere aggiunta in seguito come analisi di robustezza, senza usarla per
moltiplicare artificialmente la numerosita' dei confronti statistici.

## Uso

Controllo veloce dei dati e degli split, senza training:

```bash
python3 run_ebm.py check
```

Run di validazione (puo' richiedere diversi minuti):

```bash
python3 run_ebm.py evaluate --stage validation --run-id validation_v1
```

Solo dopo aver congelato le scelte sulla validazione, refit e benchmark H2 2025:

```bash
python3 run_ebm.py evaluate --stage test --run-id test_retrospettivo_v1 \
  --acknowledge-retrospective-test
```

Ogni run viene scritto in `out/<run-id>/` e non puo' sovrascrivere un run esistente.
Contiene punteggi, modello EBM serializzato, importanza leggibile dei termini, maschera di
valutazione, fold interni, metriche con IC bootstrap al 95%, top 10 e manifest con hash di
dati, codice e artefatti. Il file `model.pkl` va caricato solo se prodotto localmente o da
una fonte fidata. Verifica read-only:

```bash
python3 verify_run.py out/validation_v1
```

Test automatici:

```bash
python3 -m unittest discover -s tests -v
```

## Note sulle metriche

Recall, Hit rate, R-precision e MRR sono calcolate sulle pratiche incomplete. Sono
riportate sia precisione condizionata (`P@k`) sia precisione su tutte le pratiche
(`Pnc@k`). Gli intervalli bootstrap ricampionano intere pratiche. Non si applica il test
di Wilcoxon ai fold come se fossero osservazioni indipendenti.
