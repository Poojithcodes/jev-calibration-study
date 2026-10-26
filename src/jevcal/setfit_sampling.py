"""Bound stored candidate pairs without changing fixed-iteration SetFit output."""


def bounded_trainer_class(base):
    class BoundedPairTrainer(base):
        def get_dataset(self, x, y, args, max_pairs=-1):
            if args.num_iterations and args.num_iterations > 0:
                # The sampler emits iterations * n positives and negatives. Keeping
                # that prefix in each pool yields identical output, including cycling
                # when fewer distinct pairs exist. Its shuffled index array is still O(n²).
                limit = 2 * args.num_iterations * len(x)
                max_pairs = limit if max_pairs < 0 else min(max_pairs, limit)
            return super().get_dataset(x, y, args=args, max_pairs=max_pairs)
    return BoundedPairTrainer
