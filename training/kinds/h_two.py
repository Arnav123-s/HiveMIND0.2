"""The kind h_two: two release dates from PyPI, combined (hard: live facts).

A kind of task in one file (training/kinds/, docs/KINDS.md): its maker draws the task and its check from a seed, and
it has no lesson: the hive is never shown it done.
"""
import datetime
import re

from training import hive_bench as B
from training.kinds._common import _nums, _v2, _pypi, HARD_PACKAGES as PACKAGES

KIND = "h_two"
FAMILY = "hard: live facts"
HARD = True
ORDER = 30
HELD_OUT = True
NETWORK = True


def h_two(r):
    a, b = r.sample(PACKAGES, 2)

    def check(run, report):
        (_, da), (_, db) = _pypi(a), _pypi(b)
        days = abs((datetime.date.fromisoformat(da) - datetime.date.fromisoformat(db)).days)
        later = a if da >= db else b
        body = B._body(report)
        m = re.search(r"later\W+([\w.-]+)", body.lower())
        named = da == db or bool(m and m.group(1).strip(".") == later)
        # two versions and two dates are about 12 numbers; a dump of PyPI holds hundreds
        ok, tag = _v2(named and any(abs(g - days) <= 1 for g in _nums(body)), body, 16, "h_two")
        return ok, f"{a} {da}, {b} {db}: {later} is later by {days} days; in the answer: {ok}{tag}"
    return ("hard: live facts", "two facts from two sources, combined", f"Look at the latest versions of the packages {a} and {b} on PyPI. "
            "Which of the two was released more recently, and how many days lie between the two release dates? Answer in two lines: "
            "'later: NAME' and 'days: N'.", check)


make = h_two
lesson = None
