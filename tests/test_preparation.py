"""Pruebas pequeñas de las convenciones que afectarán experimentos futuros."""

import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from amazon_photo.analysis import analyze_graph
from amazon_photo.splits import make_masks, validate_masks


class PreparationTests(unittest.TestCase):
    def test_stratified_masks_and_overlap_rejection(self):
        labels = np.repeat(np.arange(8), [30, 40, 50, 60, 70, 80, 90, 100])
        masks = make_masks(labels, 42)
        again = make_masks(labels, 42)
        self.assertEqual([int(masks[k].sum()) for k in masks], [312, 104, 104])
        self.assertTrue(all(torch.equal(masks[k], again[k]) for k in masks))
        other = make_masks(labels, 43)
        self.assertFalse(torch.equal(masks["train_mask"], other["train_mask"]))
        corrupted = {k: v.clone() for k, v in masks.items()}
        test_node = torch.where(corrupted["test_mask"])[0][0]
        corrupted["train_mask"][test_node] = True
        with self.assertRaises(ValueError):
            validate_masks(corrupted, labels)

    def test_unique_edges_density_components_and_homophily(self):
        # Camino 0—1—2 y nodo aislado 3: dos pares, una arista homofílica.
        data = Data(x=torch.ones(4, 2), y=torch.tensor([0, 0, 1, 1]),
                    edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
        masks = {"train_mask": torch.tensor([True, True, False, False]),
                 "val_mask": torch.tensor([False, False, True, False]),
                 "test_mask": torch.tensor([False, False, False, True])}
        stats, _, _, _ = analyze_graph(data, masks)
        self.assertEqual(stats["num_stored_edges"], 4)
        self.assertEqual(stats["num_undirected_edges"], 2)
        self.assertAlmostEqual(stats["density"], 1 / 3)
        self.assertEqual(stats["num_components"], 2)
        self.assertEqual(stats["isolated_nodes"], 1)
        self.assertEqual(stats["edge_homophily"], .5)
        self.assertEqual(stats["degree"]["mean"], 1.)


if __name__ == "__main__":
    unittest.main()
