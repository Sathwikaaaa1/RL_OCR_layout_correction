"""Stable binary ordering IDs; existing checkpoint tensors are unchanged."""

from enum import IntEnum


class OrderingAction(IntEnum):
    RETAIN = 0
    REORDER = 1


RETAIN = OrderingAction.RETAIN
REORDER = OrderingAction.REORDER
ACTION_NAMES = {0: "RETAIN", 1: "REORDER"}
