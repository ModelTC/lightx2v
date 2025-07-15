from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
from lightx2v.utils.utils import save_videos_grid


class BaseRunner(ABC):
    """Abstract base class for all Runners

    Defines interface methods that all subclasses must implement
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def load_transformer(self):
        """Load transformer model

        Returns:
            Loaded model instance
        """
        pass

    @abstractmethod
    def load_text_encoder(self):
        """Load text encoder

        Returns:
            Text encoder instance or list of instances
        """
        pass

    @abstractmethod
    def load_image_encoder(self):
        """Load image encoder

        Returns:
            Image encoder instance
        """
        pass

    @abstractmethod
    def load_vae(self) -> Tuple[Any, Any]:
        """Load VAE encoder and decoder

        Returns:
            Tuple[vae_encoder, vae_decoder]: VAE encoder and decoder instances
        """
        pass

    @abstractmethod
    def run_image_encoder(self, img):
        """Run image encoder

        Args:
            img: Input image

        Returns:
            Image encoding result
        """
        pass

    @abstractmethod
    def run_vae_encoder(self, img):
        """Run VAE encoder

        Args:
            img: Input image

        Returns:
            VAE encoding result and additional parameters
        """
        pass

    @abstractmethod
    def run_text_encoder(self, prompt: str, img: Optional[Any] = None):
        """Run text encoder

        Args:
            prompt: Input text prompt
            img: Optional input image (for some models)

        Returns:
            Text encoding result
        """
        pass

    @abstractmethod
    def get_encoder_output_i2v(self, clip_encoder_out, vae_encode_out, text_encoder_output, img):
        """Combine encoder outputs for i2v task

        Args:
            clip_encoder_out: CLIP encoder output
            vae_encode_out: VAE encoder output
            text_encoder_output: Text encoder output
            img: Original image

        Returns:
            Combined encoder output dictionary
        """
        pass

    @abstractmethod
    def init_scheduler(self):
        """Initialize scheduler"""
        pass

    def set_target_shape(self) -> Dict[str, Any]:
        """Set target shape

        Subclasses can override this method to provide specific implementation

        Returns:
            Dictionary containing target shape information
        """
        return {}

    def save_video_func(self, images):
        """Save video implementation

        Subclasses can override this method to customize save logic

        Args:
            images: Image sequence to save
        """
        save_videos_grid(images, self.config.get("save_video_path", "./output.mp4"), n_rows=1, fps=self.config.get("fps", 8))

    def load_vae_decoder(self):
        """Load VAE decoder

        Default implementation: get decoder from load_vae method
        Subclasses can override this method to provide different loading logic

        Returns:
            VAE decoder instance
        """
        if not hasattr(self, "vae_decoder") or self.vae_decoder is None:
            _, self.vae_decoder = self.load_vae()
        return self.vae_decoder
