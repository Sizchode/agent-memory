import unittest

from experiments.recommendation_findings import source_edge_weights, support_swap_plan


class RecommendationControlsTest(unittest.TestCase):
    def test_fact_weight_and_duplicate_occurrence(self):
        facts=[("p", ("a","r","b")), ("p", ("a","r","b"))]
        self.assertEqual(source_edge_weights(facts,{"a":"ea","b":"eb"}),
                         {("ea","p"):1.5,("eb","p"):1.5})

    def test_multi_source_normalization(self):
        weights=source_edge_weights([("p",("a","r","b")),("q",("a","r","b"))],{"a":"ea","b":"eb"})
        self.assertEqual(len(weights),4)
        self.assertTrue(all(v == 1+1/3 for v in weights.values()))

    def test_self_relation_and_fact_accumulation(self):
        weights=source_edge_weights([("p",("a","r","a")),("p",("a","s","b"))],{"a":"ea","b":"eb"})
        self.assertEqual(weights[("ea","p")],2.5)
        self.assertEqual(weights[("eb","p")],1.5)

    def test_swap_does_not_remove_existing_support(self):
        plan=support_swap_plan(["g1","x","y"],["g2","z","g1"],{"g1","g2"})
        self.assertEqual(plan["position"],2)
        self.assertEqual(plan["support"],"g2")
        self.assertEqual(plan["unannotated"],"z")

    def test_unavailable_donor_not_invented(self):
        self.assertIsNone(support_swap_plan(["g1","x"],["g2","g1"],{"g1","g2"}))


if __name__ == "__main__":
    unittest.main()
