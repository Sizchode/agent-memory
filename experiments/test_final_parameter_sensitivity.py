import unittest
from experiments.final_parameter_sensitivity import SETTINGS, validate_settings


class DesignTests(unittest.TestCase):
    def test_declared_settings_change_one_factor(self):
        validate_settings(SETTINGS)

    def test_reject_joint_parameter_changes(self):
        with self.assertRaises(ValueError):
            validate_settings(dict(SETTINGS, both=(0.25, 0.01)))

    def test_reject_duplicate_reference(self):
        with self.assertRaises(ValueError):
            validate_settings(dict(SETTINGS, duplicate=(0.50, 0.05)))


if __name__ == "__main__":
    unittest.main()
