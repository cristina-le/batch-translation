import os
import sys
import time
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from app.utils import preprocess
from app.utils.preprocess import SpeakerAwareness
from app.benchmark.calculateBleu import evaluate_translation

load_dotenv()

def get_translator(version: int, **kwargs):
    """
    Get the appropriate translator based on version.
    
    Args:
        version: Translator version (1 or 2)
        **kwargs: Additional parameters for translator initialization
    
    Returns:
        Translator instance
    """
    if version == 1:
        from app.core.translator_v2 import JapaneseToEnglishTranslator
        return JapaneseToEnglishTranslator(
            temperature=kwargs.get('temperature', 0.2),
            model=kwargs.get('model', "google/gemini-2.0-flash-001"),
            context_window=kwargs.get('context_window', 3)
        )
    elif version == 2:
        from app.core.translator_v3 import JapaneseToEnglishTranslator
        return JapaneseToEnglishTranslator(
            temperature=kwargs.get('temperature', 0.05),
            model=kwargs.get('model', "openai/gpt-4o"),
            context_window=kwargs.get('context_window', 7),
            quality_threshold=kwargs.get('quality_threshold', 8.5)
        )
    else:
        raise ValueError(f"Invalid translator version: {version}. Must be 1 or 2.")

def preprocess_text(text: str) -> str:
    """Simple text preprocessing for v3."""
    return text.strip()

def apply_speaker_awareness(text: str, full_text: str, speaker_aware: bool, model: str) -> str:
    """
    Áp dụng speaker awareness cho text nếu được bật.
    """
    if not speaker_aware:
        return text
    try:
        speaker_processor = SpeakerAwareness(model)
        return speaker_processor.preprocess_speakers(text, full_text)
    except Exception as e:
        print(f"Speaker awareness error: {e}")
        return text

def translate_batch_file_v2(input_file: str, output_file: str, chunk_size: int, translator, speaker_aware: bool = False, model: str = "openai/gpt-4o"):
    """
    Ultra-optimized translation method for v2 với speaker awareness.
    """
    print(f"Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Preprocess the text
    lines = [preprocess_text(line.strip()) for line in lines if line.strip()]
    full_text = '\n'.join(lines)
    
    print(f"Total lines to translate: {len(lines)}")
    
    if speaker_aware:
        print("Speaker awareness enabled - analyzing characters...")
    
    translated_lines = []
    total_chunks = (len(lines) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i + chunk_size]
        chunk_text = '\n'.join(chunk)
        chunk_num = (i // chunk_size) + 1
        
        print(f"Translating chunk {chunk_num}/{total_chunks} ({len(chunk)} lines)...")
        
        start_time = time.time()
        try:
            # Áp dụng speaker awareness nếu được bật
            processed_text = apply_speaker_awareness(chunk_text, full_text, speaker_aware, model)
            
            translated_chunk = translator.translate(processed_text, len(chunk))
            translated_lines.extend(translated_chunk.split('\n'))
            
            elapsed = time.time() - start_time
            print(f"Chunk {chunk_num} completed in {elapsed:.2f}s")
            
        except Exception as e:
            print(f"Error translating chunk {chunk_num}: {e}")
            # Add placeholder translations to maintain line count
            translated_lines.extend([f"[Translation Error: {line}]" for line in chunk])
    
    print(f"Writing output to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in translated_lines:
            f.write(line + '\n')
    
    print(f"Translation completed! {len(translated_lines)} lines written.")
    return len(translated_lines)

def main(jp_file: str, output_file: str, en_ref_file: str = None, 
         translator_version: int = 1, chunk_size: int = 20, 
         context_window: int = 3, speaker_aware: bool = True, 
         model: str = None, temperature: float = None, 
         quality_threshold: float = 8.5):
    """
    Run batch translation from Japanese to English with selectable translator version.
    
    Args:
        jp_file: Path to Japanese input file
        output_file: Path to output file for translations
        en_ref_file: Path to English reference file for BLEU evaluation (optional)
        translator_version: Which translator version to use (1 or 2)
        chunk_size: Number of lines per chunk
        context_window: Number of previous chunks to keep in context (v1, v2)
        speaker_aware: Whether to use speaker diarization (v1, v2)
        model: Model to use for translation
        temperature: Temperature for generation
        quality_threshold: Minimum quality score to accept translation (v2)
    
    Returns:
        BLEU score and translations if en_ref_file is provided, otherwise just translations
    """
    
    # Set default models and temperatures based on version
    if model is None:
        if translator_version == 1:
            model = "google/gemini-2.0-flash-001"
        else:  # version 2
            model = "openai/gpt-4o"
    
    if temperature is None:
        if translator_version == 1:
            temperature = 0.2
        else:  # version 2
            temperature = 0.05
    
    print(f"=" * 60)
    print(f"BATCH TRANSLATION - TRANSLATOR VERSION {translator_version}")
    print(f"=" * 60)
    print(f"Settings:")
    print(f"- Translator version: {translator_version}")
    print(f"- Chunk size: {chunk_size}")
    print(f"- Model: {model}")
    print(f"- Temperature: {temperature}")
    
    if translator_version >= 1:
        print(f"- Context window: {context_window}")
    
    print(f"- Speaker awareness: {speaker_aware}")
    
    if translator_version == 2:
        print(f"- Quality threshold: {quality_threshold}")
    
    # Get the appropriate translator
    translator_kwargs = {
        'temperature': temperature,
        'model': model
    }
    
    if translator_version >= 1:
        translator_kwargs['context_window'] = context_window
    
    translator_kwargs['speaker_aware'] = speaker_aware
    
    if translator_version == 2:
        translator_kwargs['quality_threshold'] = quality_threshold
    
    translator = get_translator(translator_version, **translator_kwargs)
    
    # Handle translation based on version
    if translator_version == 2:
        # Use ultra-optimized method for v2
        if not os.path.exists(jp_file):
            print(f"Error: Input file {jp_file} not found!")
            return
        
        start_time = time.time()
        num_lines = translate_batch_file_v2(jp_file, output_file, chunk_size, translator, speaker_aware, model)
        total_time = time.time() - start_time
        
        print(f"\nTranslation Summary:")
        print(f"- Input file: {jp_file}")
        print(f"- Output file: {output_file}")
        print(f"- Lines translated: {num_lines}")
        print(f"- Total time: {total_time:.2f} seconds")
        print(f"- Average time per line: {total_time/num_lines:.2f} seconds")
        
    else:
        # Use standard method for v1
        segments = preprocess.reader(jp_file, size=chunk_size)
        
        # Đọc full text để sử dụng cho speaker awareness
        full_text = None
        if speaker_aware:
            with open(jp_file, 'r', encoding='utf-8') as f:
                full_text = f.read()
            print("Speaker awareness enabled - analyzing characters...")
        
        translations = []
        for segment in tqdm(segments):
            num_lines = len(segment.splitlines())
            
            # Áp dụng speaker awareness nếu được bật
            processed_segment = apply_speaker_awareness(segment, full_text, speaker_aware, model)
            
            translations.append(translator.translate(processed_segment, num_lines))
            time.sleep(0.5)  # Delay to avoid rate limiting

        preprocess.writer(output_file, translations)
    
    # Evaluate BLEU score if reference file is provided
    if en_ref_file and os.path.exists(en_ref_file):
        print(f"\nEvaluating against reference: {en_ref_file}")
        try:
            bleu = evaluate_translation(en_ref_file, output_file)
            print(f"BLEU score: {bleu:.4f}")
            if translator_version != 2:
                return bleu, translations
            else:
                return bleu
        except Exception as e:
            print(f"BLEU evaluation failed: {e}")
    else:
        print("No reference file available for BLEU evaluation.")
        if translator_version != 2:
            return translations
    
    print(f"\nOutput saved to: {output_file}")
    print("Translation completed successfully!")

if __name__ == "__main__":
    # Configuration - Modify these parameters as needed
    
    # File paths
    jp_file = "app/data/batch_jp.txt"
    output_file = "app/data/batch_output.txt"
    en_ref_file = "app/data/batch_en.txt"  # Set to reference file path if available
    
    # Translator version selection (1 or 2)
    translator_version = 2  # Change this to select translator version
    
    # Common parameters
    chunk_size = 20  # Number of lines per chunk
    
    # Version-specific parameters
    if translator_version == 1:
        # Version 1 (Enhanced with context and speaker awareness)
        context_window = 3  # Number of previous chunks to keep in context
        speaker_aware = False  # Enable/disable speaker diarization
        model = "anthropic/claude-sonnet-4"
        temperature = 0.2
        
        main(
            jp_file=jp_file,
            output_file=output_file,
            en_ref_file=en_ref_file,
            translator_version=translator_version,
            chunk_size=chunk_size,
            context_window=context_window,
            speaker_aware=speaker_aware,
            model=model,
            temperature=temperature
        )
        
    elif translator_version == 2:
        # Version 2 (Ultra-optimized with quality assessment)
        context_window = 3  # Larger context for better consistency
        speaker_aware = True  # Enable speaker diarization
        model = "google/gemini-2.0-flash-001"  # Best model for quality
        temperature = 0.2  # Very low for consistency
        quality_threshold = 8.5  # High quality threshold
        chunk_size = 20  # Smaller chunks for better quality
        
        main(
            jp_file=jp_file,
            output_file=output_file,
            en_ref_file=en_ref_file,
            translator_version=translator_version,
            chunk_size=chunk_size,
            context_window=context_window,
            speaker_aware=speaker_aware,
            model=model,
            temperature=temperature,
            quality_threshold=quality_threshold
        )
