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
    jp_file = "app/data/japaneseOriginal.txt"
    en_ref_file = "app/data/humanTranslation.txt"
    output_file = "app/data/translated_output.txt"

    chunk_size = 20
    model = "google/gemini-2.5-flash-preview"
    temperature = 0.3
    segments = preprocess.reader(jp_file, size=chunk_size)

    translator = JapaneseToEnglishTranslator(temperature=temperature, model=model)

    translations = []
    for segment in tqdm(segments):
        num_lines = len(segment.splitlines())
        translations.append(translator.translate(segment, num_lines))
        time.sleep(0.5)
    
    preprocess.writer(output_file, translations)
    bleu = evaluate_translation(en_ref_file, output_file)
    print(f"BLEU score: {bleu:.2f}")

if __name__ == "__main__":
    main()
