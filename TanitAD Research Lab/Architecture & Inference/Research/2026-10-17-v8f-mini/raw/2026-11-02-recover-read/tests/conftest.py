def pytest_configure(config):
    config.addinivalue_line("markers", "slow: re-runs the single-thread extractor (about 2 minutes)")
