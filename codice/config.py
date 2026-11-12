"""Configurazione unica della pipeline EBM v2."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dati"
OUT_DIR = ROOT / "out"

SEED = 20250904
INNER_FOLDS = 5
K_STRUCT = 100
K_SEMANTIC = 100
IMPACT_M_GENERAL = 40
IMPACT_M_OPERATION = 100

# Protocollo temporale canonico. Il test H2 2025 e' un benchmark retrospettivo:
# e' gia' stato consultato durante lo sviluppo della tesi e non va chiamato blind test.
DEV_END = "2025-04-01"
VALIDATION_END = "2025-07-01"
TEST_END = "2026-01-01"
# Valutazione finale: rifit su tutto il pre-2026, valutazione H1 2026 fino al 26 giugno.
# Le pratiche successive hanno meno di 90 giorni di osservazione e restano escluse.
H1_2026_END = "2026-06-27"

FEATURES = (
    "impact_doc_regime_intervento",
    "impact_doc_opere",
    "impact_doc_regime_autorizzativo",
    "impact_doc_cambio_uso",
    "doc_famiglia",
    "int_famiglia_al_deposito",
    "doc_freq_mancante",
    "doc_freq_deposito",
    "int_prior_comma_opera",
    "int_cooccorrenza_assenti",
    "int_knn",
    "int_knn_semantico_car",
    "doc_identita",
)

CATEGORICAL = (
    "doc_famiglia",
    "doc_identita",
)

# Queste colonne apprendono dalla variabile target e devono essere sempre OOF nel train.
TARGET_DERIVED = (
    "doc_freq_mancante",
    "int_prior_comma_opera",
    "int_cooccorrenza_assenti",
    "int_knn",
    "int_knn_semantico_car",
    "impact_doc_regime_intervento",
    "impact_doc_opere",
    "impact_doc_regime_autorizzativo",
    "impact_doc_cambio_uso",
)

EBM_PARAMS = {
    "max_bins": 256,
    "max_interaction_bins": 16,
    "interactions": 0,
    "learning_rate": 0.02,
    "min_samples_leaf": 5,
    "outer_bags": 14,
    "inner_bags": 0,
    "validation_size": 0.15,
    "random_state": SEED,
    "n_jobs": -1,
}
