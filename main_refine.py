import time
from dotenv import load_dotenv
from app.utils import preprocess
from app.core.refiner import TranslationRefiner
from app.benchmark.calculateBleu import evaluate_translation

load_dotenv()

def main():
    # File paths
    jp_file = "app/data/batch_jp.txt"
    en_ref_file = "app/data/batch_en.txt"
    initial_pred_file = "app/data/batch_output.txt"
    refined_pred_file = "app/data/batch_output_refined.txt"

    # 1. Read Japanese input and reference
    chunk_size = 50
    model = "google/gemini-2.5-pro-preview-03-25"
    temperature = 0.5  # Recommended temperature for refinement

    japanese_segments = preprocess.reader(jp_file, size=chunk_size)
    current_translations = preprocess.reader(initial_pred_file, size=chunk_size)

    # 2. Refine translations
    refiner = TranslationRefiner(temperature=temperature, model=model)

    refined = []
    for (jp, cur) in zip(japanese_segments, current_translations):
        num_lines = len(cur.splitlines())
        refined.append(refiner.refine(jp, cur, num_lines))
        time.sleep(0.5) # Respect API rate limits

    # 3. Write refined output
    preprocess.writer(refined_pred_file, refined)

    # 4. Evaluate BLEU score
    bleu = evaluate_translation(en_ref_file, refined_pred_file)
    print(f"Refined BLEU score: {bleu:.2f}")

if __name__ == "__main__":
    main()
