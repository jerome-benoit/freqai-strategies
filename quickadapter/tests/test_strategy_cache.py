"""Native result-cache behavior across implementation and SDK changes."""

from qa_support import REPO_ROOT, QaTestCase, model_config, temporary_directory

from scripts.backtest_cache_contract import (
    assert_native_implementation_cache,
    assert_native_startup_predictions,
)


class NativeResultCacheTest(QaTestCase):
    def test_native_startup_and_predictions_do_not_require_test_pythonpath(self):
        with temporary_directory() as directory:
            assert_native_startup_predictions(
                self, directory, model_config, REPO_ROOT / "quickadapter"
            )

    def test_implementation_changes_reject_results_but_unchanged_inputs_reuse_them(self):
        with temporary_directory() as directory:
            assert_native_implementation_cache(
                self,
                directory,
                model_config,
                "QuickAdapterV3",
                REPO_ROOT / "quickadapter",
                "from quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 import QuickAdapterRegressorV3 as CacheBase",
            )
