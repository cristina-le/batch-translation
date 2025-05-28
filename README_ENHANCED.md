# Cải Tiến Pipeline Dịch Thuật Nhật-Anh

Dự án này cung cấp một pipeline dịch thuật cải tiến từ tiếng Nhật sang tiếng Anh, tập trung vào việc giải quyết các vấn đề về mất ngữ cảnh và quan hệ chủ-khách thể trong quá trình dịch.

## Vấn Đề Được Giải Quyết

Pipeline dịch thuật ban đầu gặp phải một số vấn đề:

1. **Mất ngữ cảnh giữa các đoạn dịch**: Khi văn bản được chia thành các đoạn nhỏ để dịch, ngữ cảnh giữa các đoạn có thể bị mất, dẫn đến các lỗi dịch.

2. **Quan hệ chủ-khách thể bị đảo ngược**: Trong một số trường hợp, mối quan hệ giữa người nói và người nghe bị đảo ngược, ví dụ:
   - "I guess I can't support you when you're five years older than me after all." - Trong ngữ cảnh, người nói phải là người lớn hơn 5 tuổi, không phải người nghe.
   - "Alright, I'm going to take all my clothes off!" - Trong ngữ cảnh, người nói đang nói về việc cởi quần áo của người nghe, không phải của chính mình.

3. **Thiếu nhất quán trong việc duy trì vai trò nhân vật**: Không có thông tin rõ ràng về vai trò của người nói và người nghe.

## Giải Pháp

### 1. Cải Thiện Cơ Chế Lưu Trữ Ngữ Cảnh

- **Cửa sổ ngữ cảnh mở rộng**: Thay vì chỉ lưu trữ đoạn trước đó, pipeline mới lưu trữ nhiều đoạn trước đó (mặc định là 3 đoạn) để cung cấp ngữ cảnh rộng hơn cho quá trình dịch.

### 2. Nhận Diện Người Nói (Speaker Diarization)

- **Thêm thẻ người nói**: Tự động thêm thẻ "Female-Speaker:" vào các dòng được xác định là lời thoại của nhân vật nữ chính, giúp mô hình dịch hiểu rõ hơn về người đang nói.

### 3. Thông Tin Vai Trò Nhân Vật

- **Xác định rõ vai trò**: Cung cấp thông tin rõ ràng về vai trò của các nhân vật (người nói là nữ, người nghe là nam) trong prompt dịch.
- **Hướng dẫn về quan hệ chủ-khách thể**: Thêm hướng dẫn cụ thể về cách xử lý đại từ và quan hệ chủ-khách thể trong prompt.

## Cách Sử Dụng

### Chạy Pipeline Cải Tiến

```python
python enhanced_main.py
```

### Tùy Chỉnh Tham Số

Bạn có thể tùy chỉnh các tham số sau trong file `enhanced_main.py`:

- `chunk_size`: Kích thước đoạn (số dòng mỗi đoạn)
- `context_window`: Số đoạn trước đó giữ lại làm ngữ cảnh
- `speaker_aware`: Bật/tắt nhận diện người nói
- `model`: Mô hình dịch
- `temperature`: Nhiệt độ sinh văn bản

```python
# Các tham số có thể điều chỉnh trực tiếp ở đây
chunk_size = 20  # Kích thước đoạn (số dòng mỗi đoạn)
context_window = 3  # Số đoạn trước đó giữ lại làm ngữ cảnh
speaker_aware = True  # Bật/tắt nhận diện người nói
model = "google/gemini-2.0-flash-001"  # Mô hình dịch
temperature = 0.2  # Nhiệt độ sinh văn bản
```

### Thử Nghiệm Nhiều Cấu Hình

File `enhanced_main.py` cũng cung cấp các đoạn mã đã được comment để thử nghiệm nhiều cấu hình khác nhau:

```python
# Thử nghiệm với nhiều kích thước đoạn khác nhau
for chunk_size in [10, 20, 30]:
    print(f"\nTesting with chunk_size = {chunk_size}")
    output_file = f"app\data\ASMR/output_chunk{chunk_size}.txt"
    main(jp_file, output_file, chunk_size=chunk_size)
```

## Cải Tiến Kỹ Thuật

### EnhancedJapaneseToEnglishTranslator

Class `EnhancedJapaneseToEnglishTranslator` trong `app/core/enhanced_translator.py` cung cấp các cải tiến sau:

1. **Lưu trữ ngữ cảnh mở rộng**:
   ```python
   self.context_history = []  # Lưu trữ nhiều đoạn ngữ cảnh
   ```

2. **Nhận diện người nói**:
   ```python
   def preprocess_with_speakers(self, text: str) -> str:
       # Thêm thẻ người nói vào các dòng lời thoại
   ```

3. **Thông tin vai trò nhân vật**:
   ```python
   self.character_roles = {
       "speaker": "female",  # Người nói chính là nữ
       "listener": "male"    # Người nghe là nam
   }
   ```

4. **Prompt dịch cải tiến**:
   ```python
   CHARACTER ROLES:
   - The main speaker is a {self.character_roles["speaker"]} character.
   - The listener/protagonist is a {self.character_roles["listener"]} character.
   - CRITICAL: Maintain the correct subject-object relationships in dialogue.
   ```

## Kết Luận

Các cải tiến này giúp giải quyết vấn đề mất ngữ cảnh và quan hệ chủ-khách thể bị đảo ngược trong quá trình dịch. Bằng cách cung cấp nhiều ngữ cảnh hơn, thêm thông tin về người nói và vai trò nhân vật, pipeline dịch thuật có thể tạo ra bản dịch chính xác và tự nhiên hơn.
