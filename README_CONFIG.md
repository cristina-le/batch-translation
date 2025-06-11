# Hướng dẫn sử dụng Config cho Batch Translation

## 📁 Cấu trúc file

- `main.py`: File chính để chạy translation
- `config.py`: File chứa tất cả cấu hình
- `app/core/translator_v2.py`: Translator version 1 (tên file là v2 nhưng là version 1)
- `app/core/translator_v3.py`: Translator version 2 (ultra-optimized)

## 🚀 Cách sử dụng

### 1. Sử dụng preset có sẵn

```python
# Trong main.py
config = get_config(preset='balanced')  # Có 3 preset: 'fast', 'quality', 'balanced'
```

**Các preset:**
- `fast`: Dịch nhanh, độ chính xác thấp hơn
- `quality`: Dịch chậm nhưng chất lượng cao
- `balanced`: Cân bằng giữa tốc độ và chất lượng

### 2. Tùy chỉnh config

```python
# Override một số giá trị từ preset
