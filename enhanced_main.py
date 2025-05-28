import time
from tqdm import tqdm
from dotenv import load_dotenv
from app.utils import preprocess
from app.core.enhanced_translator import EnhancedJapaneseToEnglishTranslator
from app.benchmark.calculateBleu import evaluate_translation

load_dotenv()

def main(jp_file, output_file, en_ref_file=None, chunk_size=20, context_window=3, 
         speaker_aware=True, model="google/gemini-2.0-flash-001", temperature=0.2):
    """
    Run enhanced batch translation from Japanese to English with improved context preservation.
    
    Args:
        jp_file: Path to Japanese input file
        output_file: Path to output file for translations
        en_ref_file: Path to English reference file for BLEU evaluation (optional)
        chunk_size: Number of lines per chunk
        context_window: Number of previous chunks to keep in context
        speaker_aware: Whether to use speaker diarization
        model: Model to use for translation
        temperature: Temperature for generation
    
    Returns:
        BLEU score and translations if en_ref_file is provided, otherwise just translations
    """
    print(f"Starting enhanced translation with the following settings:")
    print(f"- Chunk size: {chunk_size}")
    print(f"- Context window: {context_window}")
    print(f"- Speaker awareness: {speaker_aware}")
    print(f"- Model: {model}")
    print(f"- Temperature: {temperature}")
    
    segments = preprocess.reader(jp_file, size=chunk_size)
    
    translator = EnhancedJapaneseToEnglishTranslator(
        temperature=temperature, 
        model=model,
        context_window=context_window,
        speaker_aware=speaker_aware
    )

    translations = []
    for segment in tqdm(segments):
        num_lines = len(segment.splitlines())
        translations.append(translator.translate(segment, num_lines))
        time.sleep(0.5)  # Delay to avoid rate limiting

    preprocess.writer(output_file, translations)
    
    if en_ref_file:
        bleu = evaluate_translation(en_ref_file, output_file)
        print(f"BLEU score: {bleu:.2f}")
        return bleu, translations
    else:
        return translations

if __name__ == "__main__":
    # Mặc định dịch file test.txt sang output.txt
    # jp_file = f"app\data\ASMR/test.txt"
    # output_file = f"app\data\ASMR/output.txt"

    jp_file = f"app/data/VN/japaneseOriginal.txt"
    output_file = f"app/data/VN/translated_output.txt"
    en_ref_file = "app/data/VN/humanTranslation.txt"
    
    # Các tham số có thể điều chỉnh trực tiếp ở đây
    chunk_size = 20  # Kích thước đoạn (số dòng mỗi đoạn)
    context_window = 3  # Số đoạn trước đó giữ lại làm ngữ cảnh
    speaker_aware = True  # Bật/tắt nhận diện người nói
    model = "google/gemini-2.0-flash-001"  # Mô hình dịch
    temperature = 0.2  # Nhiệt độ sinh văn bản
    
    main(
        jp_file=jp_file,
        output_file=output_file,
        chunk_size=chunk_size,
        context_window=context_window,
        speaker_aware=speaker_aware,
        model=model,
        temperature=temperature,
        en_ref_file=en_ref_file
    )
    
    # Để chạy với nhiều file hoặc thử nghiệm nhiều cấu hình, có thể bỏ comment các dòng dưới đây
    # và điều chỉnh theo nhu cầu
    
    # # Thử nghiệm với nhiều kích thước đoạn khác nhau
    # for chunk_size in [10, 20, 30]:
    #     print(f"\nTesting with chunk_size = {chunk_size}")
    #     output_file = f"app\data\ASMR/output_chunk{chunk_size}.txt"
    #     main(jp_file, output_file, chunk_size=chunk_size)
    
    # # Thử nghiệm với nhiều kích thước cửa sổ ngữ cảnh khác nhau
    # for context_window in [1, 3, 5]:
    #     print(f"\nTesting with context_window = {context_window}")
    #     output_file = f"app\data\ASMR/output_context{context_window}.txt"
    #     main(jp_file, output_file, context_window=context_window)
    
    # # Thử nghiệm với/không có nhận diện người nói
    # for speaker_aware in [True, False]:
    #     print(f"\nTesting with speaker_aware = {speaker_aware}")
    #     output_file = f"app\data\ASMR/output_speaker{speaker_aware}.txt"
    #     main(jp_file, output_file, speaker_aware=speaker_aware)
