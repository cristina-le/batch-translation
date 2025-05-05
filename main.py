import time
from tqdm import tqdm
from dotenv import load_dotenv
from app.utils import preprocess
from app.core.translator import JapaneseToEnglishTranslator
from app.benchmark.calculateBleu import evaluate_translation

load_dotenv()

def main():
    """
    Run batch translation from Japanese to English and evaluate BLEU score.
    """
    jp_file = "app/data/batch_jp.txt"
    en_ref_file = "app/data/batch_en.txt"
    output_file = "app/data/batch_output.txt"

    chunk_size = 20  # Original chunk size that worked well
    model = "google/gemini-2.0-flash-001"  # Keep the same model for consistency
    temperature = 0.2  # Original temperature that worked well
    segments = preprocess.reader(jp_file, size=chunk_size)

    translator = JapaneseToEnglishTranslator(temperature=temperature, model=model)

    translations = []
    for segment in tqdm(segments):
        num_lines = len(segment.splitlines())
        translations.append(translator.translate(segment, num_lines))
        time.sleep(0.5)  # Original delay
    
    preprocess.writer(output_file, translations)
    bleu = evaluate_translation(en_ref_file, output_file)
    print(f"BLEU score: {bleu:.2f}")

if __name__ == "__main__":
    main()
