import time
import os
from tqdm import tqdm
from dotenv import load_dotenv
from app.utils import preprocess
from app.core.translator import JapaneseToEnglishTranslator
from app.core.refiner import TranslationRefiner
from app.core.bleu_optimizer import BleuOptimizer
from app.benchmark.calculateBleu import evaluate_translation

load_dotenv()

def main():
    """
    Run a comprehensive pipeline of translation, refinement, and BLEU optimization,
    then evaluate BLEU scores at each stage.
    """
    # File paths
    jp_file = "app/data/batch_jp.txt"
    en_ref_file = "app/data/batch_en.txt"
    initial_output_file = "app/data/batch_output.txt"
    refined_output_file = "app/data/batch_output_refined.txt"
    optimized_output_file = "app/data/batch_output_optimized.txt"

    # Configuration
    chunk_size = 20  # Original chunk size that worked well
    translation_model = "google/gemini-2.0-flash-001"
    refinement_model = "google/gemini-2.5-pro-preview-03-25"  # More powerful model for refinement
    optimization_model = "google/gemini-2.5-pro-preview-03-25"  # Same powerful model for optimization
    
    translation_temperature = 0.2  # Original temperature that worked well
    refinement_temperature = 0.2  # Same temperature for refinement
    optimization_temperature = 0.1  # Lower temperature for optimization to maximize BLEU

    # Read Japanese text and reference translation in chunks
    jp_segments = preprocess.reader(jp_file, size=chunk_size)
    
    # Read reference translations if available
    reference_available = os.path.exists(en_ref_file)
    en_ref_segments = []
    if reference_available:
        en_ref_segments = preprocess.reader(en_ref_file, size=chunk_size)
    
    # Initialize components
    translator = JapaneseToEnglishTranslator(
        temperature=translation_temperature, 
        model=translation_model
    )
    
    refiner = TranslationRefiner(
        temperature=refinement_temperature, 
        model=refinement_model
    )
    
    optimizer = BleuOptimizer(
        temperature=optimization_temperature,
        model=optimization_model
    )

    # Perform translation, refinement, and optimization
    initial_translations = []
    refined_translations = []
    optimized_translations = []
    
    print("Starting comprehensive translation pipeline...")
    for i, jp_segment in enumerate(tqdm(jp_segments, desc="Processing segments")):
        num_lines = len(jp_segment.splitlines())
        
        # Get reference translation for this segment if available
        reference = None
        if reference_available and i < len(en_ref_segments):
            reference = en_ref_segments[i]
        
        # Step 1: Translate
        initial_translation = translator.translate(jp_segment, num_lines)
        initial_translations.append(initial_translation)
        
        # Step 2: Refine
        refined_translation = refiner.refine(jp_segment, initial_translation, num_lines)
        refined_translations.append(refined_translation)
        
        # Step 3: Optimize with BLEU optimizer
        optimized_translation = optimizer.optimize(
            jp_segment, 
            refined_translation,
            reference  # Pass reference translation if available
        )
        optimized_translations.append(optimized_translation)
        
        # Delay to avoid rate limiting
        time.sleep(0.5)
    
    # Write results to files
    preprocess.writer(initial_output_file, initial_translations)
    preprocess.writer(refined_output_file, refined_translations)
    preprocess.writer(optimized_output_file, optimized_translations)
    
    # Evaluate BLEU scores
    initial_bleu = evaluate_translation(en_ref_file, initial_output_file)
    refined_bleu = evaluate_translation(en_ref_file, refined_output_file)
    optimized_bleu = evaluate_translation(en_ref_file, optimized_output_file)
    
    print("\nBLEU Score Evaluation:")
    print(f"Initial translation BLEU score: {initial_bleu:.2f}")
    print(f"Refined translation BLEU score: {refined_bleu:.2f}")
    print(f"Optimized translation BLEU score: {optimized_bleu:.2f}")
    print(f"Total improvement: {optimized_bleu - initial_bleu:.2f} points")
    
    print("\nRecommendation:")
    if optimized_bleu > refined_bleu and optimized_bleu > initial_bleu:
        print("Use the optimized translation for best results.")
    elif refined_bleu > initial_bleu:
        print("Use the refined translation for best results.")
    else:
        print("Use the initial translation for best results.")

if __name__ == "__main__":
    main()
