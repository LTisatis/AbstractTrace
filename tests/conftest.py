def pytest_addoption(parser):
    parser.addoption("--run-live", action="store_true", help="run tests that call configured external services")
