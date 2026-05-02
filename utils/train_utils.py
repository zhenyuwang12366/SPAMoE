import math

class EarlyStopping:
    """
    Track a validation metric; stop after `patience` steps without `min_delta` improvement.
    mode='min': lower is better (e.g. val_loss).
    """
    def __init__(self, patience=20, min_delta=0.0, warmup_epochs=0, mode='min'):
        assert mode in ('min', 'max')
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.mode = mode

        self.best = math.inf if mode == 'min' else -math.inf
        self.num_bad = 0
        self.should_stop = False
        self.best_epoch = -1

    def _is_improved(self, value):
        if self.mode == 'min':
            return value < (self.best - self.min_delta)
        else:
            return value > (self.best + self.min_delta)

    def step(self, value, epoch):
        """
        Return whether training should stop (main process only). Updates best and bad-epoch count.
        """
        # Warmup: do not apply early stopping
        if epoch < self.warmup_epochs:
            return False

        if self._is_improved(value):
            self.best = value
            self.best_epoch = epoch
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
        return self.should_stop
    
def safe_random_split(dataset_size, ratios : list):
        assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"

        total = dataset_size
        raw_sizes = [r * total for r in ratios]
        sizes = [int(x) for x in raw_sizes]
        deficit = total - sum(sizes)

        # Step 1: ensure each size is at least 1 when possible
        for i in range(len(sizes)):
            if sizes[i] == 0 and deficit > 0:
                sizes[i] += 1
                deficit -= 1

        # Step 2: distribute remaining samples by largest fractional parts
        frac_with_index = sorted(
            [(raw - int(raw), i) for i, raw in enumerate(raw_sizes)],
            reverse=True
        )

        i = 0
        while deficit > 0:
            sizes[frac_with_index[i % len(sizes)][1]] += 1
            deficit -= 1
            i += 1

        assert sum(sizes) == total, "final sample count mismatch"
        train_size = sizes[0]
        val_size = sizes[1]
        return train_size, val_size