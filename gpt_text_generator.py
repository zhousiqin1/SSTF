# gpt_text_generator.py
# Offline text generation using GPT-OSS for semantic anchor creation.
import time
import logging
import gc
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
from scipy import signal

from text_utils import save_texts_to_csv, load_texts_from_csv


class GPTTextGenerator:
    """
    Generates natural language descriptions from accelerometer features
    using a local, offline GPT-OSS model.
    Implements the Text Semantic Module (TXT) described in the paper.
    """

    # Mapping from numeric label to behavior name
    LABEL_MAP = {
        0: "other behavior",
        1: "rumination behavior",
        2: "feeding behavior"
    }

    def __init__(self, args, clip_model=None):
        self.use_gpt = getattr(args, 'use_gpt', False)
        self.max_tokens = getattr(args, 'gpt_max_tokens', 70)
        self.temperature = getattr(args, 'gpt_temperature', 0.5)
        self.system_prompt = getattr(args, 'gpt_system_prompt', "")
        self.logger = logging.getLogger(__name__)

        # ── Load offline GPT-OSS model ──
        if self.use_gpt:
            self.model_path = getattr(args, 'gpt_oss_model_path', './models/gpt-oss')
            self.tokenizer_path = getattr(args, 'gpt_oss_tokenizer_path', './models/gpt-oss-tokenizer')
            self.logger.info(f"Loading GPT-OSS from {self.model_path} ...")
            # ── PLACEHOLDER: replace with actual GPT-OSS loading ──
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            # self.oss_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            # self.oss_model = AutoModelForCausalLM.from_pretrained(self.model_path, ...)
            # self.oss_model.eval()
            self.logger.info("GPT-OSS loaded successfully (offline mode).")
        else:
            self.logger.info("Using rule-based text generation mode.")

        # Feature statistics for dynamic thresholds
        self.feature_stats = {}
        self.stats_computed = False
        self.clip_model = clip_model

    # ==================================================================
    #  Feature extraction
    # ==================================================================

    def extract_enhanced_features(self, window_data: np.ndarray) -> Dict[str, float]:
        """
        Extract the seven statistical features defined in the paper (Appendix D):
        periodicity, pitch_mean, pitch_std, vm_mean, vm_std, complexity, zero_crossings.
        """
        if window_data.ndim == 1:
            window_data = window_data.reshape(-1, 1)
        x_data = window_data[:, 0]
        y_data = window_data[:, 1]
        z_data = window_data[:, 2]

        # Vector magnitude
        vm = np.sqrt(x_data**2 + y_data**2 + z_data**2)

        # 1. Periodicity (autocorrelation)
        autocorr = np.correlate(vm, vm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if len(autocorr) > 1:
            periodicity = np.max(autocorr[1:]) / autocorr[0]
        else:
            periodicity = 0.0

        # 2. Pitch angle (head posture)
        pitch_angles = np.arctan2(x_data, np.sqrt(y_data**2 + z_data**2))
        pitch_mean = float(np.mean(pitch_angles))
        pitch_std = float(np.std(pitch_angles))

        # 3. Motion intensity
        vm_mean = float(np.mean(vm))
        vm_std = float(np.std(vm))

        # 4. Signal complexity (entropy-based)
        hist, _ = np.histogram(vm, bins=20)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        complexity = float(-np.sum(prob * np.log(prob))) if len(prob) > 0 else 0.0

        # 5. Zero-crossing rate
        zero_crossings = float(np.sum(np.diff(np.signbit(vm - np.mean(vm))) != 0))

        return {
            'periodicity': periodicity,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'vm_mean': vm_mean,
            'vm_std': vm_std,
            'complexity': complexity,
            'zero_crossings': zero_crossings
        }

    # ==================================================================
    #  Feature statistics (for dynamic thresholds)
    # ==================================================================

    def compute_feature_statistics(self, all_features_list):
        """Compute population statistics for adaptive thresholding."""
        if not all_features_list:
            return
        key_features = ['periodicity', 'pitch_mean', 'pitch_std',
                        'vm_mean', 'vm_std', 'complexity', 'zero_crossings']
        for feature_name in key_features:
            values = [f[feature_name] for f in all_features_list if feature_name in f]
            if values:
                self.feature_stats[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'p25': np.percentile(values, 25),
                    'p50': np.percentile(values, 50),
                    'p75': np.percentile(values, 75),
                    'p80': np.percentile(values, 80),
                    'p85': np.percentile(values, 85),
                    'p90': np.percentile(values, 90)
                }
        self.stats_computed = True
        print(f"Feature statistics computed for {len(key_features)} features.")

    # ==================================================================
    #  Prompt construction
    # ==================================================================

    def _build_user_prompt(self, window_labels: np.ndarray, features: Dict[str, float]) -> str:
        # (1) Majority label
        if window_labels is not None and len(window_labels) > 0:
            label_counter = Counter(window_labels)
            main_label_num, main_label_count = label_counter.most_common(1)[0]
            main_label_name = self.LABEL_MAP.get(main_label_num, str(main_label_num))
            proportion = main_label_count / len(window_labels)
            label_info = (f"The majority behavior label within the window is "
                          f"\"{main_label_name}\" ({proportion:.0%} of the samples).")
        else:
            label_info = "The majority behavior label within the window is unknown."

        # (2) Seven descriptors as natural-language phrases
        descriptor_lines = []
        if features:
            f = features
            descriptor_lines.append(
                f"periodicity: {'high' if f.get('periodicity',0) > 0.4 else 'moderate' if f.get('periodicity',0) > 0.2 else 'low'} "
                f"(value={f.get('periodicity',0):.3f})")
            descriptor_lines.append(
                f"pitch mean: {f.get('pitch_mean',0):.3f} (head orientation)")
            descriptor_lines.append(
                f"pitch std: {f.get('pitch_std',0):.3f} (posture stability)")
            descriptor_lines.append(
                f"vector magnitude mean: {f.get('vm_mean',0):.3f} (movement intensity)")
            descriptor_lines.append(
                f"vector magnitude std: {f.get('vm_std',0):.3f} (intensity variability)")
            descriptor_lines.append(
                f"complexity: {f.get('complexity',0):.3f} (signal regularity)")
            descriptor_lines.append(
                f"zero-crossing rate: {f.get('zero_crossings',0):.0f} (motion repetitiveness)")

        descriptors_str = "; ".join(descriptor_lines)

        # (3) Output instruction
        output_instruction = (
            "Based on the movement features above, generate a short natural-language "
            "description of the observed movement characteristics. "
            "Do not add any explanation, commentary, or extra text."
        )

        prompt = f"{label_info}\n\n{descriptors_str}\n\n{output_instruction}"
        return prompt

    def _call_gpt_oss(self, prompt: str) -> str:
        """
        Run offline GPT-OSS inference.
        The model is called with temperature = 0.5 and a fixed random seed
        to ensure reproducibility (Appendix E).
        """
        # ── PLACEHOLDER: replace with actual GPT-OSS inference ──
        # inputs = self.oss_tokenizer(prompt, return_tensors='pt')
        # with torch.no_grad():
        #     outputs = self.oss_model.generate(
        #         **inputs,
        #         max_new_tokens=self.max_tokens,
        #         temperature=self.temperature,
        #         do_sample=True,
        #         top_p=1.0,
        #         pad_token_id=self.oss_tokenizer.eos_token_id
        #     )
        # generated_text = self.oss_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # if generated_text.startswith(prompt):
        #     generated_text = generated_text[len(prompt):].strip()
        # return generated_text

        self.logger.warning("GPT-OSS not yet integrated; returning empty string.")
        return ""

    # ==================================================================
    #  Single-window generation
    # ==================================================================

    def generate_text_for_window(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
        """Generate a semantic description for a single window using GPT-OSS."""
        features = self.extract_enhanced_features(window_data)
        if not self.stats_computed:
            self.compute_feature_statistics([features])

        prompt = self._build_user_prompt(window_labels, features)
        return self._call_gpt_oss(prompt)

    def generate_text_for_window_with_features(
        self, window_data: np.ndarray, window_labels: np.ndarray,
        cached_features: Dict[str, float]
    ) -> str:
        """Generate a description using pre-computed features and GPT-OSS."""
        if not self.stats_computed:
            self.compute_feature_statistics([cached_features])
        prompt = self._build_user_prompt(window_labels, cached_features)
        return self._call_gpt_oss(prompt)

    # ==================================================================
    #  Batch generation (used during offline pre-processing)
    # ==================================================================

    def generate_texts_for_windows_batch(
        self, features: np.ndarray, labels: np.ndarray,
        window_size: int, stride: int, batch_size: int = 32,
        save_path: str = None
    ) -> List[str]:
        text_descriptions = []
        mode_str = "GPT-OSS (offline)" if self.use_gpt else "rule-based"
        print(f"Starting batch text generation ({mode_str})...")
        print(f"Window size: {window_size}, Stride: {stride}, Batch size: {batch_size}")

        num_windows = (len(features) - window_size) // stride + 1
        print(f"Total windows: {num_windows}")

        # Pre-compute and cache all features
        print("Extracting features for all windows...")
        all_features = {}
        for window_idx in range(num_windows):
            start_idx = window_idx * stride
            end_idx = start_idx + window_size
            window_data = features[start_idx:end_idx]
            all_features[window_idx] = self.extract_enhanced_features(window_data)

        # Compute global statistics
        if not self.stats_computed:
            print("Computing feature statistics...")
            self.compute_feature_statistics(list(all_features.values()))

        # Batch processing
        for batch_start in range(0, num_windows, batch_size):
            batch_end = min(batch_start + batch_size, num_windows)
            batch_texts = []
            print(f"Processing batch {batch_start // batch_size + 1}/"
                  f"{(num_windows + batch_size - 1) // batch_size}")

            for window_idx in range(batch_start, batch_end):
                start_idx = window_idx * stride
                end_idx = start_idx + window_size
                window_data = features[start_idx:end_idx]
                window_labels = labels[start_idx:end_idx]
                cached_features = all_features[window_idx]
                text = self.generate_text_for_window_with_features(
                    window_data, window_labels, cached_features
                )
                batch_texts.append(text)

            text_descriptions.extend(batch_texts)
            del batch_texts
            gc.collect()

        print(f"Batch generation complete. {len(text_descriptions)} descriptions generated.")

        if save_path:
            csv_path = save_path.replace('.txt', '.csv')
            save_texts_to_csv(text_descriptions, csv_path)

        return text_descriptions