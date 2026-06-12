import unittest

from Common.Config.interpretability_defaults import DEFAULT_INTERPRETABILITY, get_explainer_config


class TestInterpretabilityConfig(unittest.TestCase):

    def test_defaults_are_returned_without_user_config(self):
        cfg = get_explainer_config({}, 'Lime')
        self.assertEqual(cfg['num_samples'], 5000)
        self.assertEqual(cfg['feature_selection'], 'forward_selection')

    def test_user_values_override_defaults(self):
        model_params = {
            'interpretability': {
                'Lime': {
                    'num_samples': 1000,
                    'discretizer': 'quartile'
                }
            }
        }
        cfg = get_explainer_config(model_params, 'Lime')
        self.assertEqual(cfg['num_samples'], 1000)
        self.assertEqual(cfg['discretizer'], 'quartile')
        self.assertEqual(cfg['feature_selection'], 'forward_selection')

    def test_surrogate_config_is_merged(self):
        model_params = {
            'interpretability': {
                'IntegratedGradients': {
                    'n_steps': 25,
                    'surrogate': {
                        'epochs': 50
                    }
                }
            }
        }
        cfg = get_explainer_config(model_params, 'IntegratedGradients')
        self.assertEqual(cfg['n_steps'], 25)
        self.assertEqual(cfg['surrogate']['epochs'], 50)
        self.assertEqual(cfg['surrogate']['hidden_units'], [64, 32])

    def test_all_explainers_have_defaults(self):
        for explainer in DEFAULT_INTERPRETABILITY:
            cfg = get_explainer_config({}, explainer)
            self.assertIsInstance(cfg, dict)


if __name__ == '__main__':
    unittest.main()
