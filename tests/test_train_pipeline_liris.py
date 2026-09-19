import unittest


class TrainPipelineLirisTest(unittest.TestCase):
    def test_train_pipeline_imports_with_liris_only_setup(self):
        import scene_motion_llm.train_pipeline as train_pipeline

        self.assertTrue(hasattr(train_pipeline, "parse_args"))


if __name__ == "__main__":
    unittest.main()
