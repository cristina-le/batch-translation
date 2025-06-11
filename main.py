import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
from app.utils import preprocess
from app.utils.preprocess import SpeakerAwareness
from app.benchmark.calculateBleu import evaluate_translation
from config import get_config

load_dotenv()


class TranslatorFactory:
    """Factory class to create translator suitable for version."""
    
    @staticmethod
    def create(version: int, **kwargs):
        """Create translator based on version."""
        if version == 1:
            from app.core.translator_v2 import JapaneseToEnglishTranslator
            defaults = {
                'temperature': 0.2,
                'model': "google/gemini-2.0-flash-001",
                'context_window': 3
            }
        elif version == 2:
            from app.core.translator_v3 import JapaneseToEnglishTranslator
            defaults = {
                'temperature': 0.05,
                'model': "openai/gpt-4o",
                'context_window': 7,
                'quality_threshold': 8.5
            }
        else:
            raise ValueError(f"Invalid translator version: {version}. Must be 1 or 2.")
        
        # Merge defaults with kwargs
        params = {**defaults, **kwargs}
        return JapaneseToEnglishTranslator(**params)


def apply_speaker_awareness(text: str, full_text: str, enabled: bool, model: str) -> str:
    """Apply speaker awareness if enabled."""
    if not enabled:
        return text
    try:
        processor = SpeakerAwareness(model)
        return processor.preprocess_speakers(text, full_text)
    except Exception as e:
        print(f"Speaker awareness error: {e}")
        return text


def translate_file(input_file: str, output_file: str, translator, 
                  chunk_size: int = 20, speaker_aware: bool = False):
    """
    Translate file from Japanese to English.
    
    Args:
        input_file: Input file path
        output_file: Output file path
        translator: Translator instance
        chunk_size: Number of lines per chunk
        speaker_aware: Enable/disable speaker awareness
    """
    print(f"Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    full_text = '\n'.join(lines) if speaker_aware else None
    
    print(f"Total lines to translate: {len(lines)}")
    if speaker_aware:
        print("Speaker awareness enabled - analyzing characters...")
    
    translated_lines = []
    total_chunks = (len(lines) + chunk_size - 1) // chunk_size
    
    # Translate each chunk
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i + chunk_size]
        chunk_text = '\n'.join(chunk)
        chunk_num = (i // chunk_size) + 1
        
        print(f"Translating chunk {chunk_num}/{total_chunks} ({len(chunk)} lines)...")
        
        start_time = time.time()
        try:
            # Apply speaker awareness if needed
            if speaker_aware:
                chunk_text = apply_speaker_awareness(
                    chunk_text, full_text, True, translator.model
                )
            
            # Translate chunk
            translated = translator.translate(chunk_text, len(chunk))
            translated_lines.extend(translated.split('\n'))
            
            elapsed = time.time() - start_time
            print(f"Chunk {chunk_num} completed in {elapsed:.2f}s")
            
        except Exception as e:
            print(f"Error translating chunk {chunk_num}: {e}")
            # Add placeholder to maintain line count
            translated_lines.extend([f"[Translation Error: {line}]" for line in chunk])
    
    # Write results
    print(f"Writing output to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in translated_lines:
            f.write(line + '\n')
    
    print(f"Translation completed! {len(translated_lines)} lines written.")
    return translated_lines


def evaluate_bleu(reference_file: str, output_file: str) -> float:
    """Evaluate BLEU score if reference file is available."""
    if not reference_file or not os.path.exists(reference_file):
        print("No reference file available for BLEU evaluation.")
        return None
    
    print(f"\nEvaluating against reference: {reference_file}")
    try:
        bleu = evaluate_translation(reference_file, output_file)
        print(f"BLEU score: {bleu:.4f}")
        return bleu
    except Exception as e:
        print(f"BLEU evaluation failed: {e}")
        return None


def main(jp_file: str, 
         output_file: str, 
         en_ref_file: str = None,
         translator_version: int = 2,
         chunk_size: int = 20,
         speaker_aware: bool = False,
         **translator_kwargs):
    """
    Run batch translation from Japanese to English.
    
    Args:
        jp_file: Japanese input file
        output_file: Output file
        en_ref_file: English reference file (optional)
        translator_version: Translator version (1 or 2)
        chunk_size: Number of lines per chunk
        speaker_aware: Enable/disable speaker awareness
        **translator_kwargs: Other parameters for translator
    """
    # Check input file
    if not os.path.exists(jp_file):
        print(f"Error: Input file {jp_file} not found!")
        return
    
    # Print configuration info
    print(f"{'=' * 60}")
    print(f"BATCH TRANSLATION - VERSION {translator_version}")
    print(f"{'=' * 60}")
    print(f"Settings:")
    print(f"- Input: {jp_file}")
    print(f"- Output: {output_file}")
    print(f"- Chunk size: {chunk_size}")
    print(f"- Speaker awareness: {speaker_aware}")
    
    # Create translator
    translator = TranslatorFactory.create(translator_version, **translator_kwargs)
    print(f"- Model: {translator.model}")
    print(f"- Temperature: {translator.temperature}")
    print(f"- Context window: {translator.context_window}")
    
    if translator_version == 2 and hasattr(translator, 'quality_threshold'):
        print(f"- Quality threshold: {translator.quality_threshold}")
    
    # Perform translation
    start_time = time.time()
    translations = translate_file(
        input_file=jp_file,
        output_file=output_file,
        translator=translator,
        chunk_size=chunk_size,
        speaker_aware=speaker_aware
    )
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\nTranslation Summary:")
    print(f"- Total time: {total_time:.2f} seconds")
    print(f"- Average time per line: {total_time/len(translations):.2f} seconds")
    
    # Evaluate BLEU if available
    bleu_score = evaluate_bleu(en_ref_file, output_file)
    
    print(f"\nOutput saved to: {output_file}")
    print("Translation completed successfully!")
    
    return translations, bleu_score


if __name__ == "__main__":

    config = get_config(preset='balanced')
        
    # Run translation
    main(**config)
