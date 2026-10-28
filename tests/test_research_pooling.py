import numpy as np
import pandas as pd

from ima.pool_selection import _meeting_log_loss


def test_nested_selection_objective_weights_meetings_equally():
    rows = []
    for date, losses, prefix in (("2020-01-01", [0.1], "a"),
                                 ("2020-01-02", [1.0, 1.0, 1.0], "b")):
        for index, loss in enumerate(losses):
            winner_probability = float(np.exp(-loss))
            rows.extend([
                {"date": pd.Timestamp(date), "race_id": f"{prefix}-{index}",
                 "target_probability": 1.0},
                {"date": pd.Timestamp(date), "race_id": f"{prefix}-{index}",
                 "target_probability": 0.0},
            ])
    frame = pd.DataFrame(rows)
    probabilities = np.asarray([
        np.exp(-0.1), 1-np.exp(-0.1),
        np.exp(-1.0), 1-np.exp(-1.0),
        np.exp(-1.0), 1-np.exp(-1.0),
        np.exp(-1.0), 1-np.exp(-1.0),
    ])
    assert np.isclose(_meeting_log_loss(probabilities, frame), 0.55)
