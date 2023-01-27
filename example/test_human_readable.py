import datetime

import human_readable


def test_human_readable() -> None:
    timedelta = datetime.timedelta(days=2, hours=10, seconds=1)

    assert human_readable.times.time_delta(timedelta) == "2 days"
    assert (
        human_readable.times.precise_delta(timedelta)
        == "2 days, 10 hours and 1 second"
    )
