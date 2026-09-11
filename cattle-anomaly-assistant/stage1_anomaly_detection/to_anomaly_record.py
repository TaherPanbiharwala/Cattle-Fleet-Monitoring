"""Public library entry point for M1d's safe daily fusion contract."""

from .behavior_classifier.context import BehaviorDailyContext, CusumDailyResult, fuse_daily_records

__all__ = ["BehaviorDailyContext", "CusumDailyResult", "fuse_daily_records"]
