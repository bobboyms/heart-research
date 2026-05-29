from __future__ import annotations


from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


REPO_ROOT = SCRIPT_DIR.parents[1]


DEFAULT_TCN_CHECKPOINT = (
    REPO_ROOT / "modeling" / "Grupo E TCN segmentacao frame a frame" / "outputs_noncausal_overlap" / "best_model.pt"
)


DEFAULT_PREDICTED_TSV_DIR = (
    REPO_ROOT
    / "feature extraction"
    / "Grupo B v2 features relativas por local com TCN predito"
    / "outputs"
    / "predicted_tsvs"
)


LABEL_SYSTOLE = 2


LABEL_DIASTOLE = 4


PHASE_LABELS = {
    "systole": (LABEL_SYSTOLE,),
    "diastole": (LABEL_DIASTOLE,),
    "both": (LABEL_SYSTOLE, LABEL_DIASTOLE),
}


LOCATION_ORDER = ("AV", "PV", "TV", "MV")


@dataclass(frozen=True)
class StftConfig:
    target_sample_rate: int
    n_fft: int
    hop_length: int
    high_hz: float
    max_frames: int
    min_systole_seconds: float
    systole_threshold: float | None = None
    systole_margin_ms: float = 0.0
    low_hz: float = 0.0
    cnn_phase_mode: str = "systole"
    spectrogram_type: str = "stft"
    n_mels: int = 64
    stft_segment_mode: str = "concat"
    use_ground_truth_segments: bool = False
    use_temporal_features: bool = False
    window_mode: str = "phase"
    peak_window_seconds: float = 1.0
    # Re-reference the systole spectrogram by the same recording's diastole baseline per frequency
    # (C[f,t] = systole_logmag - median_t diastole_logmag) to expose the systolic energy excess.
    phase_contrast: bool = False
    # Dual-channel: stack [systole, contrast] along the frequency axis -> (2*freq, T) so the encoder
    # sees both the raw systole texture and the diastole-referenced contrast.
    phase_contrast_dual: bool = False
    # Robust contrast: divide by the diastole MAD per frequency (robust z-score), per Grupo B v3.1.
    phase_contrast_robust: bool = False


@dataclass(frozen=True)
class ModelConfig:
    freq_bins: int
    max_frames: int
    base_channels: int
    dropout: float
    dilations: tuple[int, ...]
    pooling: str
    encoder_block: str = "residual"
    n_temporal_features: int = 0
    arch: str = "cnn"
    rnn_hidden: int = 64
    rnn_layers: int = 2
    rnn_type: str = "gru"
    # freq2d encoder (Conv2d keeping the frequency axis)
    freq_emphasis: bool = False
    freq_attention: bool = False
    freq_low_hz: float = 100.0
    freq_high_hz: float = 600.0
    freq_emphasis_alpha_init: float = 0.0
    freq_sample_rate: int = 4000
    freq_fmax: float = 1000.0
    freq_mel_scale: bool = False
    # parallel branch over the murmur frequency band (for the cnn 1D arch)
    freq_linear_branch: bool = False
    freq_linear_hidden: int = 32
    freq_linear_arch: str = "transformer"  # "transformer" or "mlp"
    freq_linear_heads: int = 4
    freq_linear_layers: int = 2
    # auxiliary multi-task head predicting systolic murmur pitch (Low/Medium/High) from the pooled
    # encoder features. 0 disables. Supervised only on Present recordings (Tier 2 multi-task).
    aux_pitch_classes: int = 0


N_TEMPORAL_FEATURES = 12


@dataclass(frozen=True)
class RecordingItem:
    recording_id: str
    patient_id: str
    location: str
    wav_path: Path
    tsv_path: Path
    murmur: str
    recording_present: bool
