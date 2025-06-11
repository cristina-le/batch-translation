# Config Usage Guide for Batch Translation

## 📁 File Structure

- `main.py`: Main file to run translation
- `config.py`: File containing all configurations
- `app/core/translator_v2.py`: Translator version 1 (file name is v2 but it's version 1)
- `app/core/translator_v3.py`: Translator version 2 (ultra-optimized)

## 🚀 Usage

### 1. Using available presets

```python
# In main.py
config = get_config(preset='balanced')  # 3 presets available: 'fast', 'quality', 'balanced'
```

**Presets:**
- `fast`: Fast translation, lower accuracy
- `quality`: Slow translation but high quality
- `balanced`: Balance between speed and quality

### 2. Customizing config

```python
# Override some values from preset
