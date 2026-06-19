"""Test that runs network verification as a hard gate."""


def test_network_verification():
    from data.verify import run_verification

    result = run_verification()
    assert result.passed, f"Network verification failed: {result.summary}"
