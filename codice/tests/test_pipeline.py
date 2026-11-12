import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
import feature2
import split
import testo
import universo
import valuta


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real = universo.definitivo(feature2.carica(config.DATA_DIR))

    def test_schema_has_identity_and_not_instructor(self):
        self.assertEqual(len(config.FEATURES), 13)
        self.assertIn("doc_identita", config.FEATURES)
        self.assertNotIn("int_prior_istruttore", config.FEATURES)
        self.assertFalse(any(name.startswith("pr_") for name in config.FEATURES))
        for name in (
            "impact_doc_regime_intervento",
            "impact_doc_opere",
            "impact_doc_regime_autorizzativo",
            "impact_doc_cambio_uso",
        ):
            self.assertIn(name, config.FEATURES)
            self.assertIn(name, config.TARGET_DERIVED)
        self.assertIn("doc_freq_mancante", config.TARGET_DERIVED)
        self.assertNotIn("pr_paesaggistica", config.CATEGORICAL)
        self.assertEqual(config.EBM_PARAMS["interactions"], 0)

    def test_real_data_counts_and_temporal_protocol(self):
        D = self.real
        self.assertEqual((D["npr"], D["nd"]), (4103, 149))
        masks = split.temporal_splits(D["P"])
        expected = {
            "development": (1592, 598),
            "validation": (596, 212),
            "final_train": (2188, 810),
            "test": (1126, 406),
            "refit_2026": (3314, 1216),
            "h1_2026": (740, 251),
            "excluded_future": (49, 7),
        }
        for name, (n, incomplete) in expected.items():
            self.assertEqual(int(masks[name].sum()), n)
            self.assertEqual(int(D["TAR"][masks[name]].any(1).sum()), incomplete)

    def test_fold_unit_is_unique_practice(self):
        D = self.real
        train = split.temporal_splits(D["P"])["development"]
        fold_ids = split.inner_fold_ids(
            train, D["TAR"].any(1), D["P"].Codice_Pratica.to_numpy(), 5, config.SEED
        )
        self.assertTrue(np.all(fold_ids[~train] == -1))
        self.assertEqual(set(np.unique(fold_ids[train])), set(range(5)))

    def test_deposited_documents_never_ranked(self):
        scores = np.array([[0.1, 0.99, 0.8, 0.7], [0.9, 0.8, 0.7, 0.6]])
        deposited = np.array([[False, True, False, False], [True, True, False, False]])
        order = valuta.top_k(scores, deposited, k=2, seed=1)
        self.assertFalse(np.take_along_axis(deposited, order, axis=1).any())

    def test_missing_frequency_is_recomputed_oof(self):
        n, nd = 30, 4
        P = pd.DataFrame({
            "Codice_Pratica": [f"P{i:03d}" for i in range(n)],
            "comma": np.where(np.arange(n) % 2, "23", "22c1"),
            "opera": np.where(np.arange(n) % 3, "OP1", "OP2"),
            "tipo_intervento": "T",
            "assenso": "non_necessari",
            "cambio_uso": "nessuno",
            "paesaggistica": False,
            "n_opere": 1,
        })
        dep = np.zeros((n, nd), bool)
        dep[np.arange(n), np.arange(n) % nd] = True
        target = np.zeros((n, nd), bool)
        target[np.arange(n), (np.arange(n) + 1) % nd] = np.arange(n) % 3 == 0
        opere = pd.DataFrame({
            "Codice_Pratica": P.Codice_Pratica,
            "Cod_Opera": np.where(np.arange(n) % 3, "OP1", "OP2"),
        })
        D = {"P": P, "DEP": dep, "TAR": target, "fam": np.array(["A", "A", "B", "B"]),
             "npr": n, "nd": nd, "opere": opere}
        train = np.ones(n, bool)
        folds = np.arange(n) % 5
        columns = feature2.costruisci_out_of_fold(D, train, folds)
        for fold in range(5):
            held = folds == fold
            expected = target[~held].mean(0)
            np.testing.assert_allclose(
                columns["doc_freq_mancante"][held],
                np.tile(expected, (int(held.sum()), 1)),
            )

    def test_impact_encodings_are_candidate_specific_and_oof(self):
        n, nd = 30, 3
        P = pd.DataFrame({
            "Codice_Pratica": [f"I{i:03d}" for i in range(n)],
            "comma": np.where(np.arange(n) % 2, "23", "22c1"),
            "opera": np.where(np.arange(n) % 3, "OP1", "OP2"),
            "tipo_intervento": np.where(np.arange(n) % 2, "T1", "T2"),
            "assenso": np.where(np.arange(n) % 4, "non_necessari", "richiesta"),
            "cambio_uso": np.where(np.arange(n) % 5, "non_dichiarato", "nessuno"),
            "paesaggistica": np.arange(n) % 7 == 0,
            "n_opere": 1,
        })
        dep = np.zeros((n, nd), dtype=bool)
        target = np.zeros((n, nd), dtype=bool)
        target[np.arange(n) % 2 == 0, 0] = True
        target[np.arange(n) % 2 == 1, 1] = True
        target[np.arange(n) % 3 == 0, 2] = True
        opere = pd.DataFrame({
            "Codice_Pratica": P.Codice_Pratica,
            "Cod_Opera": P.opera,
        })
        D = {
            "P": P,
            "DEP": dep,
            "TAR": target,
            "fam": np.array(["A", "B", "C"]),
            "npr": n,
            "nd": nd,
            "opere": opere,
        }
        train = np.ones(n, dtype=bool)
        folds = np.arange(n) % 5
        original = feature2.costruisci_out_of_fold(D, train, folds)
        self.assertGreater(
            np.ptp(original["impact_doc_regime_intervento"][0]), 0
        )

        changed_D = dict(D)
        changed_target = target.copy()
        held = folds == 0
        changed_target[held] = ~changed_target[held]
        changed_D["TAR"] = changed_target
        changed = feature2.costruisci_out_of_fold(changed_D, train, folds)
        for name in (
            "impact_doc_regime_intervento",
            "impact_doc_opere",
            "impact_doc_regime_autorizzativo",
            "impact_doc_cambio_uso",
        ):
            np.testing.assert_allclose(
                original[name][held], changed[name][held], rtol=0, atol=0
            )

    def test_semantic_training_rows_ignore_test_text(self):
        n, nd = 20, 3
        train = np.arange(n) < 15
        folds = np.full(n, -1)
        folds[train] = np.arange(15) % 5
        target = np.zeros((n, nd), bool)
        target[np.arange(n), np.arange(n) % nd] = True
        D = {"TAR": target, "npr": n}
        base = np.array([f"ristrutturazione facciata edificio gruppo {i % 4}" for i in range(n)], object)
        changed = base.copy()
        changed[~train] = "testo completamente alieno xyzq"
        a = testo.colonna_out_of_fold(D, train, base, folds)
        b = testo.colonna_out_of_fold(D, train, changed, folds)
        np.testing.assert_allclose(a[train], b[train], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
