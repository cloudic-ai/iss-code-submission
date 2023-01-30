class DataFolderFull(Exception):
    def __init__(self, average_image_size: int):
        self.average_image_size = average_image_size


class ExecutionTimeExceeded(Exception):
    pass
