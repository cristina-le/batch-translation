import time
from tqdm import tqdm
from dotenv import load_dotenv
from app.utils import preprocess
from app.core.translator import JapaneseToEnglishTranslator
from app.benchmark.calculateBleu import evaluate_translation


load_dotenv()

def main():
    # File paths
    jp_file = "app/data/batch_jp.txt"
    en_ref_file = "app/data/batch_en.txt"
    output_file = "app/data/batch_output.txt"

    # 1. Read and chunk Japanese input
    chunk_size = 20
    model = "google/gemini-2.0-flash-001"
    temperature = 0.3  # Recommended temperature for initial translation
    segments = preprocess.reader(jp_file, size=chunk_size)

    # 2. Translate each chunk
    translator = JapaneseToEnglishTranslator(temperature=temperature, model=model)

    translations = []
    for segment in tqdm(segments):
        num_lines = len(segment.splitlines())
        translations.append(translator.translate(segment, num_lines))
        time.sleep(0.5)  # Respect API rate limits
    
    preprocess.writer(output_file, translations)
    # 3. Evaluate BLEU score
    bleu = evaluate_translation(en_ref_file, output_file)
    print(f"BLEU score: {bleu:.2f}")

if __name__ == "__main__":
    main()
