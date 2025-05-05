import time
from tqdm import tqdm
from dotenv import load_dotenv
from app.utils import preprocess
from app.core.translator import JapaneseToEnglishTranslator
from app.core.refiner import TranslationRefiner
from app.benchmark.calculateBleu import evaluate_translation

load_dotenv()

def main():
    """
    Run a combined pipeline of translation and refinement, then evaluate BLEU scores.
    """
    # File paths
    jp_file = "app/data/VN/japaneseOriginal.txt"
    en_ref_file = "app/data/VN/humanTranslation.txt"
    initial_output_file = "app/data/VN/translated_output.txt"
    refined_output_file = "app/data/VN/translated_output_refined.txt"

    # Configuration
    chunk_size = 20  # Original chunk size that worked well
    translation_model = "google/gemini-2.0-flash-001"
    refinement_model = "google/gemini-2.0-flash-001"
    translation_temperature = 0.3  # Original temperature that worked well
    refinement_temperature = 0.3  # Same temperature for refinement

    # Read Japanese text in chunks
    segments = preprocess.reader(jp_file, size=chunk_size)
    
    # Initialize translator and refiner
    translator = JapaneseToEnglishTranslator(
        temperature=translation_temperature, 
        model=translation_model
    )
    
    refiner = TranslationRefiner(
        temperature=refinement_temperature, 
        model=refinement_model
    )

    # Perform translation and refinement
    initial_translations = []
    refined_translations = []
    
    print("Starting translation and refinement pipeline...")
    for segment in tqdm(segments, desc="Processing segments"):
        num_lines = len(segment.splitlines())
        
        # Step 1: Translate
        initial_translation = translator.translate(segment, num_lines)
        initial_translations.append(initial_translation)
        
        # Step 2: Refine
        refined_translation = refiner.refine(segment, initial_translation, num_lines)
        refined_translations.append(refined_translation)
        
        # Delay to avoid rate limiting
        time.sleep(0.5)
    
    # Write results to files
    preprocess.writer(initial_output_file, initial_translations)
    preprocess.writer(refined_output_file, refined_translations)
    
    # Evaluate BLEU scores
    initial_bleu = evaluate_translation(en_ref_file, initial_output_file)
    refined_bleu = evaluate_translation(en_ref_file, refined_output_file)
    
    print(f"Initial translation BLEU score: {initial_bleu:.2f}")
    print(f"Refined translation BLEU score: {refined_bleu:.2f}")
    print(f"Improvement: {refined_bleu - initial_bleu:.2f} points")

if __name__ == "__main__":
    main()
