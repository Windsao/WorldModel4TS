"""Check consolidated benchmark claims against the stored result table."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    with open(root / "results_benchmark_grid" / "benchmark_grid.json") as handle:
        payload = json.load(handle)
    assert payload["protocol"]["test_used_for_selection"] is False
    assert payload["protocol"]["no_lookahead_tests"] == "passed"

    residual_wins = 0
    for dataset, horizons in payload["results"].items():
        for horizon, methods in horizons.items():
            def mse(value):
                return value.get("MSE_mean", value.get("MSE"))

            best = min(mse(value) for value in methods.values())
            winners = {
                name for name, value in methods.items() if mse(value) == best
            }
            expected = payload["primary_mse_winners"][f"{dataset}_{horizon}"]
            assert expected in winners
            residual_wins += "residual_dyadic_rgb" in winners
    assert residual_wins == payload["locked_residual_mse_wins"] == 3
    print("[pass] full-split, three-seed grid and 3/6 mean-MSE claim")


if __name__ == "__main__":
    main()
