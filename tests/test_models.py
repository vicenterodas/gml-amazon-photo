"""La MLP ignora el grafo y la GCN depende de las conexiones disponibles."""

import unittest

import torch
from torch.nn import functional as F

from amazon_photo.models import GCN, MLP, build_classifier


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[1., 0., 0.],
                               [0., 1., 0.],
                               [1., 1., 0.],
                               [0., 0., 1.]])
        self.edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.other_edges = torch.tensor([[0, 3, 3, 0], [3, 0, 0, 3]])

    def test_logits_shape_and_finite_values(self):
        for kind in ("mlp", "gcn"):
            model = build_classifier(kind, 3, 8, 2)
            model.eval()
            logits = model(self.x, self.edges)
            self.assertEqual(tuple(logits.shape), (4, 2))
            self.assertTrue(torch.isfinite(logits).all())

    def test_mlp_ignores_edges_and_gcn_uses_them(self):
        torch.manual_seed(0)
        mlp = MLP(3, 8, 2)
        torch.manual_seed(0)
        gcn = GCN(3, 8, 2)
        mlp.eval()
        gcn.eval()
        self.assertTrue(torch.equal(mlp(self.x), mlp(self.x, self.other_edges)))
        self.assertTrue(torch.equal(mlp(self.x, self.edges), mlp(self.x, self.other_edges)))
        self.assertFalse(torch.equal(gcn(self.x, self.edges), gcn(self.x, self.other_edges)))

    def test_gcn_accepts_graph_without_edges(self):
        model = GCN(3, 8, 2)
        model.eval()
        empty = torch.empty((2, 0), dtype=torch.long)
        logits = model(self.x, empty)
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_eval_is_deterministic(self):
        model = GCN(3, 8, 2, dropout=0.9)
        model.eval()
        first = model(self.x, self.edges)
        second = model(self.x, self.edges)
        self.assertTrue(torch.equal(first, second))

    def test_same_parameter_count(self):
        mlp = build_classifier("mlp", 745, 64, 8)
        gcn = build_classifier("gcn", 745, 64, 8)
        mlp_count = sum(parameter.numel() for parameter in mlp.parameters())
        gcn_count = sum(parameter.numel() for parameter in gcn.parameters())
        self.assertEqual(mlp_count, gcn_count)

    def test_reset_follows_initialization_seed(self):
        model = MLP(3, 4, 2)
        torch.manual_seed(1)
        model.reset_parameters()
        first = [parameter.detach().clone() for parameter in model.parameters()]
        torch.manual_seed(1)
        model.reset_parameters()
        for previous, current in zip(first, model.parameters()):
            self.assertTrue(torch.equal(previous, current))
        torch.manual_seed(2)
        model.reset_parameters()
        changed = [not torch.equal(previous, current)
                   for previous, current in zip(first, model.parameters())]
        self.assertTrue(any(changed))

    def test_masked_loss_reaches_parameters(self):
        model = GCN(3, 4, 2, dropout=0.0)
        logits = model(self.x, self.edges)
        mask = torch.tensor([True, False, False, True])
        labels = torch.tensor([0, 1, 1, 1])
        F.cross_entropy(logits[mask], labels[mask]).backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_unknown_classifier_is_rejected(self):
        with self.assertRaises(ValueError):
            build_classifier("gat", 3, 8, 2)


if __name__ == "__main__":
    unittest.main()
