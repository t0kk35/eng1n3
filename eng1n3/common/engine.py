"""
Common abstract class for all engines.
(c) 2023 3ngin3
"""
from abc import ABC

import logging


class EngineContextException(Exception):
    def __init__(self, message: str):
        super().__init__('Error creating Engine context; ' + message)


# Keep Original LogRecordFactory. We're going to overwrite it.
ORIGINAL_LOG_FACTORY = logging.getLogRecordFactory()


class EngineContext(ABC):
    """
    Base class for engine creation. Implemented a context for future use. All engines will be implemented a context
    in order to be able to provide data and create/keep/destroy connections and resources.
    """
    def __init__(self, no_logging=False):
        if not no_logging:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s.%(msecs)03d %(trunc_name)-30s %(levelname)-8s %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
            # Override the LogRecordFactory with a truncating version
            logging.setLogRecordFactory(self.log_record_factory)
            logger = logging.getLogger(__name__)
            logger.info('Start Engine...')

    def __enter__(self) -> 'EngineContext':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @staticmethod
    def log_record_factory(*args, **kwargs):
        record = ORIGINAL_LOG_FACTORY(*args, **kwargs)
        name = record.name
        record.trunc_name = f'{name[:1]}...{name[-25:]}' if len(name) > 30 else name
        return record
