"""LLM prompt generation and local text generation utilities."""

import logging
import torch

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LLMExplanationGenerator:
    """Generate textual explanations from motion summaries and sentiment predictions."""

    def __init__(self,
                 model_name='microsoft/phi-2-mini',
                 use_local_llm=False,
                 max_length=128):
        self.model_name = model_name
        self.use_local_llm = use_local_llm
        self.max_length = max_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline = None

        if self.use_local_llm and TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map='auto' if self.device == 'cuda' else None,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                )
                self.pipeline = pipeline(
                    'text-generation',
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == 'cuda' else -1,
                    max_length=self.max_length,
                    do_sample=False
                )
                logger.info(f"Loaded local LLM model: {self.model_name} on {self.device}")
            except Exception as e:
                logger.warning(f"Could not load LLM model {self.model_name}: {e}")
                self.pipeline = None
        elif self.use_local_llm:
            logger.warning("Transformers is not installed; LLM generation is disabled.")

    def build_prompt(self,
                     sentiment,
                     confidence,
                     motion_summary,
                     motion_metrics,
                     frame_count):
        """Build a motion-aware prompt for LLM processing."""
        prompt = (
            f"A video analysis report is provided. The model predicted sentiment '{sentiment}' with confidence "
            f"{confidence:.2%}. The motion summary is: {motion_summary}. "
            f"Motion metrics: mean magnitude {motion_metrics['mean_magnitude']:.2f}, "
            f"max magnitude {motion_metrics['max_magnitude']:.2f}, "
            f"standard deviation {motion_metrics['std_magnitude']:.2f}, "
            f"and motion coverage {motion_metrics['motion_count']} frames out of {frame_count}. "
            "Produce a concise, human-readable explanation of the visual and motion cues "
            "that support the sentiment prediction."
        )
        return prompt

    def generate_explanation(self,
                             sentiment,
                             confidence,
                             motion_summary,
                             motion_metrics,
                             frame_count):
        """Generate an explanation using an LLM or rule-based fallback."""
        prompt = self.build_prompt(
            sentiment,
            confidence,
            motion_summary,
            motion_metrics,
            frame_count
        )

        if self.pipeline is not None:
            try:
                output = self.pipeline(prompt, max_length=self.max_length, num_return_sequences=1)
                text = output[0]['generated_text']
                return text.replace(prompt, '').strip() or self.rule_based_explanation(
                    sentiment, confidence, motion_summary, motion_metrics, frame_count
                )
            except Exception as e:
                logger.warning(f"LLM generation error: {e}")

        return self.rule_based_explanation(
            sentiment,
            confidence,
            motion_summary,
            motion_metrics,
            frame_count
        )

    def rule_based_explanation(self,
                               sentiment,
                               confidence,
                               motion_summary,
                               motion_metrics,
                               frame_count):
        """Fallback explanation generator if a local LLM is unavailable."""
        explanation = (
            f"The video contains {frame_count} analyzed frames. "
            f"Motion analysis indicates a mean magnitude of {motion_metrics['mean_magnitude']:.2f} "
            f"and a maximum magnitude of {motion_metrics['max_magnitude']:.2f}. "
            f"The motion pattern is described as: {motion_summary}. "
            f"The predicted sentiment is {sentiment} with confidence {confidence:.2%}. "
        )

        if sentiment == 'Positive':
            explanation += (
                "The motion dynamics appear lively and stable, suggesting a positive emotional tone."
            )
        elif sentiment == 'Neutral':
            explanation += (
                "The motion is moderate and balanced, producing a neutral sentiment classification."
            )
        else:
            explanation += (
                "The motion is intense and variable, supporting a negative sentiment outcome."
            )

        return explanation
