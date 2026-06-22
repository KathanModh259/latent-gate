"""
Multi-Language Support — Non-English text compression.

Provides language detection and translation capabilities for
processing text in multiple languages.

Features:
  - Automatic language detection
  - Translation to English for processing
  - Language-specific compression prompts
  - Support for major languages
"""

import logging
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


logger = logging.getLogger("latent_gate.multilang")


# ============================================================================
# Language Codes
# ============================================================================

LANGUAGE_CODES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "el": "Greek",
    "he": "Hebrew",
    "ro": "Romanian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Filipino",
    "sw": "Swahili",
}


# ============================================================================
# Language Detection
# ============================================================================

@dataclass
class LanguageInfo:
    """Information about detected language."""
    code: str
    name: str
    confidence: float
    is_english: bool


def detect_language(text: str) -> LanguageInfo:
    """
    Detect the language of input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        LanguageInfo with detected language details
    """
    # Try using langdetect if available
    try:
        from langdetect import detect, detect_langs
        
        lang_code = detect(text)
        probs = detect_langs(text)
        confidence = probs[0].prob if probs else 0.5
        
        return LanguageInfo(
            code=lang_code,
            name=LANGUAGE_CODES.get(lang_code, lang_code),
            confidence=confidence,
            is_english=lang_code == "en",
        )
    except ImportError:
        pass
    
    # Fallback: simple heuristic-based detection
    return _heuristic_detection(text)


def _heuristic_detection(text: str) -> LanguageInfo:
    """
    Simple heuristic-based language detection.
    
    This is a fallback when langdetect is not installed.
    """
    text_lower = text.lower()
    
    # Common English words
    english_words = {"the", "is", "are", "was", "were", "have", "has", "had", "be", "been",
                     "do", "does", "did", "will", "would", "could", "should", "may", "might",
                     "can", "shall", "must", "need", "dare", "ought", "used"}
    
    # Common Spanish words
    spanish_words = {"el", "la", "los", "las", "un", "una", "es", "son", "está", "están",
                     "de", "del", "en", "con", "por", "para", "que", "como", "pero", "más"}
    
    # Common French words
    french_words = {"le", "la", "les", "un", "une", "est", "sont", "été", "être", "avoir",
                    "de", "du", "des", "en", "dans", "pour", "par", "sur", "avec", "que"}
    
    # Common German words
    german_words = {"der", "die", "das", "ein", "eine", "ist", "sind", "war", "haben", "sein",
                    "von", "zu", "in", "mit", "auf", "für", "an", "nach", "bei", "über"}
    
    # Common Chinese characters (simplified)
    chinese_chars = set("的一是不了人我在有他这中大来上个国到说们为子和你地出会也时要就可以")
    
    # Common Japanese hiragana/katakana
    japanese_chars = set("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめも")
    
    # Common Korean characters
    korean_chars = set("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")
    
    # Common Arabic characters
    arabic_chars = set("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
    
    # Common Hindi characters (Devanagari)
    hindi_chars = set("अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
    
    # Count matches
    words = set(text_lower.split())
    
    scores = {
        "en": len(words & english_words),
        "es": len(words & spanish_words),
        "fr": len(words & french_words),
        "de": len(words & german_words),
    }
    
    # Check character-based languages
    char_counts = {
        "zh": sum(1 for c in text if c in chinese_chars),
        "ja": sum(1 for c in text if c in japanese_chars),
        "ko": sum(1 for c in text if c in korean_chars),
        "ar": sum(1 for c in text if c in arabic_chars),
        "hi": sum(1 for c in text if c in hindi_chars),
    }
    
    # Find best match
    all_scores = {**scores, **char_counts}
    best_lang = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_lang]
    
    # Calculate confidence
    total = sum(all_scores.values())
    confidence = best_score / max(total, 1) if total > 0 else 0.5
    
    return LanguageInfo(
        code=best_lang,
        name=LANGUAGE_CODES.get(best_lang, best_lang),
        confidence=min(confidence, 0.9),  # Cap at 0.9 for heuristic
        is_english=best_lang == "en",
    )


# ============================================================================
# Translation
# ============================================================================

def translate_to_english(
    text: str,
    source_lang: str,
    config=None,
) -> str:
    """
    Translate text to English for processing.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        config: Optional PipelineConfig for Ollama settings
        
    Returns:
        Translated text in English
    """
    if source_lang == "en":
        return text
    
    # Try using Ollama for translation
    try:
        from latent_gate.config import PipelineConfig
        from latent_gate.fast_client import FastClient
        
        if config is None:
            config = PipelineConfig()
        
        client = FastClient(config)
        
        prompt = f"""Translate the following text from {LANGUAGE_CODES.get(source_lang, source_lang)} to English. 
Preserve the original meaning and context. Return ONLY the translation, no explanations.

Text to translate:
{text[:2000]}"""
        
        translation = client.ollama_generate(
            model=config.predictor_model,
            prompt=prompt,
            max_tokens=1000,
        )
        
        client.close()
        return translation.strip()
        
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text


# ============================================================================
# Language-Aware Prompts
# ============================================================================

LANGUAGE_PROMPTS = {
    "en": {
        "compress": "Extract the essential semantics from this English text.",
        "summarize": "Summarize this English conversation concisely.",
    },
    "es": {
        "compress": "Extrae la semántica esencial de este texto en español.",
        "summarize": "Resume esta conversación en español de manera concisa.",
    },
    "fr": {
        "compress": "Extrayez la sémantique essentielle de ce texte en français.",
        "summarize": "Résumez cette conversation en français de manière concise.",
    },
    "de": {
        "compress": "Extrahieren Sie die wesentliche Semantik dieses deutschen Textes.",
        "summarize": "Fassen Sie dieses deutsche Gespräch kurz zusammen.",
    },
    "zh": {
        "compress": "提取此中文文本的基本语义。",
        "summarize": "简洁地总结此中文对话。",
    },
    "ja": {
        "compress": "この日本語テキストの本質的な意味を抽出してください。",
        "summarize": "この日本語の会話を簡潔に要約してください。",
    },
    "ko": {
        "compress": "이 한국어 텍스트의 본질적인 의미를 추출하세요.",
        "summarize": "이 한국어 대화를 간결하게 요약하세요.",
    },
}


def get_language_prompt(language_code: str, prompt_type: str = "compress") -> Optional[str]:
    """
    Get a language-specific prompt.
    
    Args:
        language_code: Language code (e.g., "en", "es", "zh")
        prompt_type: Prompt type ("compress" or "summarize")
        
    Returns:
        Language-specific prompt or None
    """
    lang_prompts = LANGUAGE_PROMPTS.get(language_code, {})
    return lang_prompts.get(prompt_type)


# ============================================================================
# Multi-Language Text Processor
# ============================================================================

class MultiLanguageProcessor:
    """
    Text processor with multi-language support.
    
    Automatically detects language and processes text accordingly.
    
    Usage:
        processor = MultiLanguageProcessor(config)
        result = processor.process("Texto en español")
    """
    
    def __init__(self, config=None, translate_to_en: bool = True):
        """
        Initialize multi-language processor.
        
        Args:
            config: Optional PipelineConfig
            translate_to_en: Whether to translate non-English text to English
        """
        from latent_gate.config import PipelineConfig
        self.config = config or PipelineConfig()
        self.translate_to_en = translate_to_en
    
    def process(self, text: str, **kwargs) -> Tuple[str, LanguageInfo]:
        """
        Process text with language detection and optional translation.
        
        Args:
            text: Input text
            **kwargs: Additional arguments for processing
            
        Returns:
            Tuple of (processed_text, language_info)
        """
        # Detect language
        lang_info = detect_language(text)
        logger.info(f"Detected language: {lang_info.name} ({lang_info.code}) "
                    f"with {lang_info.confidence:.2%} confidence")
        
        # Translate if needed
        if not lang_info.is_english and self.translate_to_en:
            logger.info(f"Translating from {lang_info.name} to English")
            text = translate_to_english(text, lang_info.code, self.config)
        
        return text, lang_info
    
    def get_prompt(self, language_code: str, prompt_type: str = "compress") -> str:
        """
        Get appropriate prompt for the language.
        
        Args:
            language_code: Language code
            prompt_type: Prompt type
            
        Returns:
            Prompt string
        """
        prompt = get_language_prompt(language_code, prompt_type)
        if prompt:
            return prompt
        
        # Fallback to English prompt
        from latent_gate.text_processor import TextProcessor
        return TextProcessor.COMPRESS_PROMPT


# ============================================================================
# Convenience Functions
# ============================================================================

def detect_text_language(text: str) -> str:
    """
    Detect the language of input text.
    
    Args:
        text: Input text
        
    Returns:
        Language code (e.g., "en", "es", "zh")
    """
    return detect_language(text).code


def is_english(text: str) -> bool:
    """
    Check if text is in English.
    
    Args:
        text: Input text
        
    Returns:
        True if English, False otherwise
    """
    return detect_language(text).is_english


def get_supported_languages() -> Dict[str, str]:
    """
    Get list of supported languages.
    
    Returns:
        Dictionary mapping language codes to names
    """
    return LANGUAGE_CODES.copy()
