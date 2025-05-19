import time
from tqdm import tqdm
from dotenv import load_dotenv
from app.utils import preprocess
from app.core.translator import JapaneseToEnglishTranslator
from app.benchmark.calculateBleu import evaluate_translation

load_dotenv()

def main(jp_file, output_file, en_ref_file=None):
    """
    Run batch translation from Japanese to English and evaluate BLEU score.
    """
    # jp_file = "app/data/batch_jp.txt"
    # en_ref_file = "app/data/batch_en.txt"
    # output_file = "app/data/batch_output.txt"
    # jp_file = "app/data/VN/japaneseOriginal.txt"
    # en_ref_file = "app/data/VN/humanTranslation.txt"
    # output_file = "app/data/VN/translated_output.txt"

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
    if en_ref_file:
        bleu = evaluate_translation(en_ref_file, output_file)
        print(f"BLEU score: {bleu:.2f}")
        return bleu, translations
    else:
        return translations

if __name__ == "__main__":
    jp_file = f"app\data\ASMR/test.txt"
    output_file = f"app\data\ASMR/output.txt"
    main(jp_file, output_file)
    # for i in range(1,10):
    #     jp_file = f"app\data/test\JP\japaneseOriginal_part{i}.txt"
    #     en_ref_file = f"app\data\VN\humanTranslation.txt"
    #     output_file = f"app\data\VN/translated_output.txt"
    #     best_bleu = evaluate_translation(en_ref_file, output_file)
    #     print(best_bleu)
    # best_translate = ""
    # for j in range (5):
    #     print(f"Attempt {j}:")
    #     current_bleu, translations = main(jp_file, en_ref_file, output_file)
    #     if current_bleu > best_bleu:
    #         best_bleu = current_bleu
    #         best_translate = translations
    # print(f"Best bleu: {best_bleu}")
    # preprocess.writer(output_file, best_translate)

