class BaseDataset:
    """Common surface for the project's dataset wrappers.

    Subclasses describe a dataset by setting the class attributes below
    and implementing the three loaders. Construction defers the actual
    data load: ``__init__`` only marks the dataset as not-yet-loaded
    (``self.dataset = None``) so subclasses can lazily call
    :meth:`download` the first time a loader runs.
    """

    dataset_type = None
    path = None
    name = None
    subsets = []
    keys = []

    def __init__(self, *args, **kwargs):
        # Sentinel consulted by subclass loaders, which call download()
        # on first use when this is still None.
        self.dataset = None

    def download(self):
        raise NotImplementedError(
            f"download not implemented for {self.__class__.__name__}"
        )

    def batchify(self, dataset, batch_size=64):
        """Slice ``dataset`` into a list of column-dict mini-batches.

        Reads only the columns named in ``self.keys`` and chops each one
        into contiguous spans of at most ``batch_size`` rows, returning a
        list of ``{key: column_slice}`` dicts in row order.
        """
        column_keys = self.keys
        columns = {key: dataset[key] for key in column_keys}
        n_rows = len(columns[column_keys[0]])
        batches = []
        for start in range(0, n_rows, batch_size):
            stop = start + batch_size
            batches.append({key: columns[key][start:stop] for key in column_keys})
        return batches

    def load_dataset_for_eval(self):
        raise NotImplementedError(
            f"load_dataset_for_eval not implemented for {self.__class__.__name__}"
        )

    def load_dataset_for_classification(self):
        raise NotImplementedError(
            f"load_dataset_for_classification not implemented for "
            f"{self.__class__.__name__}"
        )
