import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class RerankerModelStore:
    _model = None
    _is_loading = False

    @classmethod
    def load_model(cls):
        if cls._model is not None:
            return

        if cls._is_loading:
             logger.warning("Model is already loading in another thread.")
             return

        cls._is_loading = True
        logger.info("Loading Reranker Model (Global)...")
        try:
            cls._model = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)
            logger.info("Reranker Model Loaded Successfully!")
        except Exception as e:
            logger.critical(f"Failed to load Reranker Model: {e}")
            cls._is_loading = False
            raise e
        finally:
            cls._is_loading = False

    @classmethod
    def get_model(cls):
        return cls._model