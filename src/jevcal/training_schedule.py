"""Resolve a minimum-update rule into whole training epochs."""
import math


def training_schedule(examples, effective_batch, epochs=10, min_updates=0):
    if examples <= 0 or effective_batch <= 0 or epochs <= 0 or min_updates < 0:
        raise ValueError('Invalid training schedule inputs')
    updates_per_epoch = math.ceil(examples / effective_batch)
    resolved_epochs = max(epochs, math.ceil(min_updates / updates_per_epoch))
    return resolved_epochs, resolved_epochs * updates_per_epoch
