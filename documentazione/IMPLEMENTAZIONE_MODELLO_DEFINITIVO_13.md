# Implementazione definitiva — EBM additivo a 13 variabili

## Decisione implementata

La pipeline principale utilizza ora un Explainable Boosting Machine con
`interactions=0` e 13 variabili. Le otto variabili `pr_*`, costanti fra i
documenti candidati della stessa pratica, non appartengono più allo schema del
modello. Rimangono ricostruibili in `feature2.py` esclusivamente per la
compatibilità con le analisi storiche.

Le quattro nuove variabili sono:

1. `impact_doc_regime_intervento`;
2. `impact_doc_opere`;
3. `impact_doc_regime_autorizzativo`;
4. `impact_doc_cambio_uso`.

Ogni impact è definito come:

`m-estimate(P(target | documento assente, contesto)) - P(target | documento assente)`.

Si usa `m=40` per i contesti ordinari e `m=100` per le opere. Nelle pratiche
con più opere viene conservato il massimo impact tra le singole opere. Il cambio
d'uso non dichiarato assume valore zero. Tutti i valori del training sono
costruiti out-of-fold con gli stessi cinque fold delle altre feature
supervisionate.

## File modificati

- `config.py`: schema definitivo di 13 feature, impact fra le variabili
  supervisionate, EBM additivo e parametri di attenuazione;
- `feature2.py`: implementazione canonica dei quattro impact e conservazione
  delle associazioni complete pratica–opera;
- `run_ebm.py`: manifest versione 2 con contratto esplicito degli impact;
- `tests/test_pipeline.py`: controlli sullo schema a 13 variabili, sulla
  specificità per candidato e sull'assenza di leakage OOF;
- `README.md`: descrizione aggiornata del modello definitivo.

## Verifiche eseguite

- compilazione Python dei moduli modificati: superata;
- 7 test automatici: superati;
- confronto con la matrice sperimentale approvata: uguaglianza esatta,
  differenza assoluta massima pari a zero;
- run di validazione e run finale H1 2026: completati;
- verifica di hash, schema, artefatti e metriche tramite `verify_run.py`:
  `status: ok`.

## Risultati

### Risultato finale: primo semestre 2026

Rifit su 3.314 pratiche precedenti al 1° gennaio 2026, valutazione su 740 pratiche
presentate dal 1° gennaio al 26 giugno 2026, delle quali 251 incomplete. Estrazione
dati art. 22/23 aggiornata ad agosto 2026.

| Metrica | Valore | IC 95% |
|---|---:|---:|
| R@3 | 58,49% | 53,17–63,79 |
| Hit@3 | 73,71% | 68,35–79,09 |
| R@5 | 66,22% | 61,29–71,21 |
| Hit@5 | 80,08% | 75,00–85,14 |
| R@10 | 73,80% | 68,95–77,97 |
| Hit@10 | 85,66% | 81,12–89,78 |
| MRR | 72,03% | 67,05–76,77 |
| R-precision | 54,39% | 49,15–59,50 |

### Validazione: secondo trimestre 2025 (fase di sviluppo)

Run usato per scegliere la configurazione, calcolato sullo snapshot dati precedente
(settembre 2026): 1.584 pratiche di sviluppo e 590 di validazione, delle quali 209
incomplete. Non è il risultato finale.

| Metrica | Valore |
|---|---:|
| R@3 | 53,64% |
| Hit@3 | 72,25% |
| R@5 | 58,69% |
| Hit@5 | 74,16% |
| R@10 | 70,63% |
| Hit@10 | 81,34% |
| MRR | 67,50% |
| R-precision | 47,43% |
