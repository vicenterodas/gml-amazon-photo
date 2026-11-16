"""Controles de fuga de etiquetas, selección y eliminación de conexiones."""

import unittest

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.utils import is_undirected

from amazon_photo.models import GCN, MLP
from amazon_photo.perturbation import edge_subsets, graph_statistics
from amazon_photo.training import TrainingConfig, classification_metrics, fit, seed_everything, train_epoch


class PerturbationTests(unittest.TestCase):
    def setUp(self):
        pairs = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 4]])
        self.edges = torch.cat((pairs, pairs.flip(0)), dim=1)

    def test_pairs_are_removed_together_and_subsets_are_nested(self):
        original = self.edges.clone()
        graphs = edge_subsets(self.edges, 6, [0, .2, .6, 1], 42)
        previous = set(map(tuple, graphs[0].T.tolist()))
        for rate, expected_pairs in ((0, 5), (.2, 4), (.6, 2), (1, 0)):
            edges = graphs[rate]
            current = set(map(tuple, edges.T.tolist()))
            self.assertTrue(current <= previous)
            self.assertEqual(len(current), expected_pairs * 2)
            self.assertTrue(is_undirected(edges, num_nodes=6))
            previous = current
        self.assertTrue(torch.equal(original, self.edges))
        self.assertTrue(torch.equal(graphs[0], original))
        self.assertEqual(graph_statistics(graphs[1], 6)["components"], 6)
        self.assertEqual(graph_statistics(graphs[1], 6)["isolated_nodes"], 6)

    def test_same_seed_replays_graph_without_changing_global_rng(self):
        np.random.seed(5)
        expected = np.random.random()
        np.random.seed(5)
        first = edge_subsets(self.edges, 6, [.6], 42)
        self.assertEqual(np.random.random(), expected)
        second = edge_subsets(self.edges, 6, [.6], 42)
        self.assertTrue(torch.equal(first[.6], second[.6]))

    def test_invalid_graphs_and_rates_are_rejected(self):
        for rate in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                edge_subsets(self.edges, 6, [rate], 42)
        with self.assertRaises(ValueError):
            edge_subsets(torch.tensor([[0], [1]]), 2, [.5], 42)
        with self.assertRaises(ValueError):
            edge_subsets(torch.tensor([[0], [0]]), 2, [.5], 42)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., 0.]])
        self.edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.train_mask = torch.tensor([True, True, False, False])
        self.val_mask = torch.tensor([False, False, True, True])
        self.y = torch.tensor([0, 1, 1, 0])

    def test_nontraining_labels_cannot_affect_gradient_update(self):
        states = []
        for labels in (self.y, torch.tensor([0, 1, -999, -999])):
            seed_everything(42)
            model = GCN(2, 4, 2, dropout=0.)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            train_epoch(model, optimizer, self.x, self.edges,
                        self.train_mask, labels[self.train_mask])
            states.append({k: v.clone() for k, v in model.state_dict().items()})
        self.assertTrue(all(torch.equal(states[0][k], states[1][k]) for k in states[0]))

    def test_checkpoint_is_chosen_by_validation_loss_and_run_repeats(self):
        states, histories = [], []
        for _ in range(2):
            seed_everything(43)
            model = MLP(2, 4, 2, dropout=.2)
            history, metadata = fit(model, self.x, self.edges, self.train_mask,
                                    self.y[self.train_mask], self.val_mask, self.y[self.val_mask],
                                    TrainingConfig(epochs=15, patience=4, learning_rate=.2))
            minimum = min(row["val_loss"] for row in history)
            self.assertAlmostEqual(metadata["best_val_loss"], minimum)
            self.assertEqual(metadata["best_epoch"], next(r["epoch"] for r in history if r["val_loss"] == minimum))
            with torch.no_grad():
                restored = F.cross_entropy(model(self.x, self.edges)[self.val_mask], self.y[self.val_mask])
            self.assertAlmostEqual(float(restored), minimum)
            self.assertFalse(model.training)
            histories.append(history)
            states.append({k: v.clone() for k, v in model.state_dict().items()})
        self.assertEqual(histories[0], histories[1])
        self.assertTrue(all(torch.equal(states[0][k], states[1][k]) for k in states[0]))

    def test_ties_keep_first_checkpoint_and_trigger_patience(self):
        seed_everything(0)
        model = MLP(2, 4, 2, dropout=0.)
        history, metadata = fit(model, self.x, self.edges, self.train_mask, self.y[self.train_mask],
                                self.val_mask, self.y[self.val_mask],
                                TrainingConfig(epochs=20, patience=3, learning_rate=1e-30, weight_decay=0.))
        self.assertEqual(metadata["best_epoch"], 1)
        self.assertEqual(len(history), 4)

    def test_empty_graph_gcn_matches_mlp_with_same_weights(self):
        mlp = MLP(2, 4, 2, dropout=0.)
        gcn = GCN(2, 4, 2, dropout=0.)
        with torch.no_grad():
            gcn.conv1.lin.weight.copy_(mlp.lin1.weight)
            gcn.conv1.bias.copy_(mlp.lin1.bias)
            gcn.conv2.lin.weight.copy_(mlp.lin2.weight)
            gcn.conv2.bias.copy_(mlp.lin2.bias)
        mlp.eval()
        gcn.eval()
        self.assertTrue(torch.allclose(mlp(self.x), gcn(self.x, torch.empty((2, 0), dtype=torch.long))))

    def test_macro_f1_includes_unpredicted_classes(self):
        scores = classification_metrics(torch.tensor([[3., 0.], [3., 0.]]), torch.tensor([0, 1]), 2)
        self.assertEqual(scores["accuracy"], .5)
        self.assertAlmostEqual(scores["macro_f1"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
