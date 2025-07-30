<!-- At the top after badges -->
<div align="center">
  <img src="https://raw.githubusercontent.com/AGo444/SRT/master/icon-96.png" alt="WhisperX Translator Icon" width="96" height="96">
  
  # WhisperX Subtitle Creator and Translator
  
  *🏠 100% Local AI Translation | 🚀 RTX 50 Series Optimized*
</div>

A high-performance Docker container that automatically generates English subtitles from video files using WhisperX and translates them to multiple languages using Hugging Face translation models. Specifically optimized for NVIDIA RTX 50 series GPUs with CUDA 12.8 support.

## 🚀 Features

### Core Functionality
- **🎧 High-Quality Transcription**: Uses OpenAI WhisperX with advanced alignment
- **🌐 Multi-Language Translation**: Supports 10 languages with Helsinki-NLP models
- **🏠 100% Local Processing**: All AI inference runs on your RTX 5070 Ti
- **⚡ GPU Acceleration**: Optimized for RTX 40/50 series with CUDA 12.8
- **🎯 Perfect Timing**: Preserves exact subtitle timing during translation
- **📁 Batch Processing**: Processes entire directories automatically
- **🔄 Resume Support**: Skips already processed files
- **🔒 Complete Privacy**: No data ever leaves your Unraid server

### Technical Features
- **🤖 Helsinki-NLP Models**: High-quality OPUS-MT translation models running locally
- **💾 Smart Caching**: Persistent model storage for instant offline operation
- **🔍 Word-Level Alignment**: Precise timing with alignment models
- **📊 High Accuracy Mode**: Full precision float32 processing
- **🛡️ Error Recovery**: Robust error handling and timeout protection
- **🧹 Clean Output**: Suppressible library warnings for cleaner logs
- **⚡ RTX 50 Optimized**: Special optimizations for Ada Lovelace architecture

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA RTX 30/40/50 series (recommended: RTX 5070 Ti or higher)
- **VRAM**: Minimum 8GB, recommended 12GB+ (RTX 5070 Ti: 16GB)
- **RAM**: 16GB system RAM recommended
- **Storage**: SSD recommended for model cache

### Software
- Docker with NVIDIA Container Runtime
- NVIDIA Driver 525+ (for CUDA 12.8)
- Unraid 6.12+ (for Unraid users)

## 🔧 Installation

### Unraid Template Installation

1. **Download Template**:
   ```
   https://raw.githubusercontent.com/AGo444/SRT/main/my-whisperx-translator-gpu.xml
   ```

2. **Add Repository**: In Unraid Community Applications
   - Go to Docker tab → Add Container
   - Template: `https://raw.githubusercontent.com/AGo444/SRT/main/my-whisperx-translator-gpu.xml`

3. **Configure Paths**:
   - **Input Directory**: `/mnt/user/YourVideoShare/` → Your video files location
   - **Cache Directory**: `/mnt/user/appdata/whisperx/cache` → Model cache storage

### Docker Compose

```yaml
version: '3.8'
services:
  whisperx-translator:
    image: agoddrie/whisperx-translator:1.6.3
    container_name: whisperx-subtitle-translator
    runtime: nvidia
    mem_limit: 12g
    memswap_limit: 16g
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - TARGET_LANGUAGE=nl
      - WHISPERX_MODEL=large-v3
      - DEBUG=0
      - SUPPRESS_LIBRARY_WARNINGS=1
      - HIGH_ACCURACY_MODE=1
      - ENABLE_WORD_ALIGNMENT=1
      - BATCH_PROCESSING=1
      - GPU_OPTIMIZATION=1
      - GPU_PRESET=rtx5070ti
      - RTX50_OPTIMIZED=1
      - BATCH_SIZE=4
      - COMPUTE_TYPE=float32
      - TF32_ENABLED=0
      - DETERMINISTIC_MODE=1
    volumes:
      - /path/to/your/videos:/data:rw
      - /path/to/cache:/config/cache:rw
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "python3 -c 'import torch; print(\"GPU available:\", torch.cuda.is_available())' || exit 1"]
      interval: 60s
      timeout: 15s
      retries: 3
      start_period: 30s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Docker Run Command

```bash
docker run -d \
  --name="whisperx-subtitle-translator" \
  --runtime=nvidia \
  --gpus all \
  --restart=unless-stopped \
  --memory=12g \
  --memory-swap=16g \
  --health-cmd="python3 -c 'import torch; print(\"GPU available:\", torch.cuda.is_available())' || exit 1" \
  --health-interval=60s \
  --health-timeout=15s \
  --health-retries=3 \
  --health-start-period=30s \
  -e TARGET_LANGUAGE="nl" \
  -e WHISPERX_MODEL="large-v3" \
  -e DEBUG="0" \
  -e SUPPRESS_LIBRARY_WARNINGS="1" \
  -e HIGH_ACCURACY_MODE="1" \
  -e ENABLE_WORD_ALIGNMENT="1" \
  -e BATCH_PROCESSING="1" \
  -e GPU_OPTIMIZATION="1" \
  -e GPU_PRESET="rtx5070ti" \
  -e RTX50_OPTIMIZED="1" \
  -e BATCH_SIZE="4" \
  -e COMPUTE_TYPE="float32" \
  -e TF32_ENABLED="0" \
  -e DETERMINISTIC_MODE="1" \
  -v "/path/to/videos:/data:rw" \
  -v "/path/to/cache:/config/cache:rw" \
  agoddrie/whisperx-translator:1.6.3
```

## ⚙️ Configuration

### Basic Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_LANGUAGE` | `nl` | Target language (nl, de, fr, es, it, pt, ru, ja, ko, zh) |
| `WHISPERX_MODEL` | `large-v3` | WhisperX model (large-v3, large-v2, medium, small) |
| `DEBUG` | `0` | Enable debug logging (0/1) |
| `SUPPRESS_LIBRARY_WARNINGS` | `1` | Hide deprecation warnings (0/1) |
| `BATCH_SIZE` | `4` | Processing batch size |

### GPU Optimization Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_PRESET` | `auto` | GPU model preset (auto, rtx5070ti, rtx4090, etc.) |
| `RTX50_OPTIMIZED` | `1` | Enable RTX 50 series optimizations (0/1) |
| `TF32_ENABLED` | `0` | Enable TF32 for speed vs accuracy (0/1) |
| `GPU_OPTIMIZATION` | `1` | General GPU optimizations (0/1) |

### Accuracy Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HIGH_ACCURACY_MODE` | `1` | Enable maximum accuracy (0/1) |
| `ENABLE_WORD_ALIGNMENT` | `1` | Precise word-level timing (0/1) |
| `VERIFY_TIMING` | `1` | Verify timing preservation (0/1) |
| `COMPUTE_TYPE` | `float32` | Precision (float32/float16/int8) |
| `TEMPERATURE` | `0.0` | Model temperature (0.0 = most accurate) |
| `BEAM_SIZE` | `5` | Beam search size |
| `BEST_OF` | `5` | Number of candidates to generate |

### Advanced Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VAD_FILTER` | `1` | Voice Activity Detection (0/1) |
| `AUDIO_QUALITY` | `high` | Audio preprocessing quality |
| `PREVENT_TEXT_SPLITTING` | `1` | Prevent text splitting during translation |
| `TRANSLATION_DELAY` | `0.1` | Rate limiting between translations |
| `TRANSLATION_TIMEOUT` | `15` | Translation request timeout (seconds) |
| `DETERMINISTIC_MODE` | `1` | Consistent results (0/1) |

## 🎯 Usage

### File Processing
The container automatically processes all video files in the mounted `/data` directory:

```
/data/
├── Movie1.mkv              → Movie1.en.srt + Movie1.nl.srt
├── Series/
│   ├── S01E01.mp4         → S01E01.en.srt + S01E01.nl.srt
│   └── S01E02.mp4         → S01E02.en.srt + S01E02.nl.srt
└── Documentary.avi         → Documentary.en.srt + Documentary.nl.srt
```

### Supported Formats
- **Video**: `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`
- **Output**: `.srt` subtitle files

### Languages Supported

| Code | Language | Hugging Face Model |
|------|----------|-------------------|
| `nl` | Dutch | Helsinki-NLP/opus-mt-en-nl |
| `de` | German | Helsinki-NLP/opus-mt-en-de |
| `fr` | French | Helsinki-NLP/opus-mt-en-fr |
| `es` | Spanish | Helsinki-NLP/opus-mt-en-es |
| `it` | Italian | Helsinki-NLP/opus-mt-en-it |
| `pt` | Portuguese | Helsinki-NLP/opus-mt-en-pt |
| `ru` | Russian | Helsinki-NLP/opus-mt-en-ru |
| `ja` | Japanese | Helsinki-NLP/opus-mt-en-ja |
| `ko` | Korean | Helsinki-NLP/opus-mt-en-ko |
| `zh` | Chinese | Helsinki-NLP/opus-mt-en-zh |

## 📊 Performance

### Benchmark Results (RTX 5070 Ti 16GB)

| Model | Accuracy | Speed | VRAM Usage | Processing Time (1h video) |
|-------|----------|-------|------------|---------------------------|
| `large-v3` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 12-14GB | 10-15 minutes |
| `large-v2` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 10-12GB | 8-12 minutes |
| `medium` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 6-8GB | 5-8 minutes |
| `small` | ⭐⭐ | ⭐⭐⭐⭐⭐ | 4-6GB | 3-5 minutes |

### Processing Times (1 hour video)

| GPU Model | VRAM | Recommended Model | Processing Time | Batch Size |
|-----------|------|-------------------|----------------|------------|
| **RTX 5090** | 32GB | large-v3 | 6-10 minutes | 16 |
| **RTX 5080** | 16GB | large-v3 | 8-12 minutes | 12 |
| **RTX 5070 Ti** | 16GB | large-v3 | 10-15 minutes | 8 |
| **RTX 5070** | 12GB | large-v3 | 12-18 minutes | 6 |
| **RTX 4090** | 24GB | large-v3 | 8-12 minutes | 12 |
| **RTX 4070 Ti** | 12GB | large-v3 | 12-18 minutes | 6 |
| **RTX 4060** | 8GB | medium | 20-30 minutes | 4 |
| **RTX 3080** | 10GB | large-v2 | 15-25 minutes | 4 |

### RTX 50 Series Optimizations

The container is specifically optimized for RTX 50 series GPUs with:

- **CUDA 12.8 Support** - Latest CUDA toolkit for RTX 50 series
- **cuDNN 9.11.0** - Optimized deep learning libraries
- **PyTorch Nightly** - Bleeding-edge RTX 50 series support
- **Memory Management** - Optimized VRAM allocation for 16GB+ cards
- **Batch Processing** - Leverages high memory bandwidth
- **Health Monitoring** - Container health checks

#### RTX 5070 Ti Optimal Settings:
```yaml
environment:
  - GPU_PRESET=rtx5070ti
  - WHISPERX_MODEL=large-v3
  - BATCH_SIZE=8
  - HIGH_ACCURACY_MODE=1
  - ENABLE_WORD_ALIGNMENT=1
  - COMPUTE_TYPE=float32
  - TF32_ENABLED=0  # Disabled for maximum accuracy
  - RTX50_OPTIMIZED=1
```

## 🔍 Monitoring

### View Logs
```bash
# Clean output (default with SUPPRESS_LIBRARY_WARNINGS=1)
docker logs -f whisperx-subtitle-translator

# Full debug output
docker logs -f whisperx-subtitle-translator --tail 50
```

### Health Check
```bash
# Check container health
docker inspect whisperx-subtitle-translator | grep -A 10 '"Health"'

# Manual health test
docker exec whisperx-subtitle-translator python3 -c "
import torch
print(f'GPU Available: {torch.cuda.is_available()}')
print(f'GPU Device: {torch.cuda.get_device_name(0)}')
print(f'CUDA Version: {torch.version.cuda}')
print('✅ Container healthy!')
"
```

### Performance Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 5

# Monitor container resources
docker stats whisperx-subtitle-translator

# Check processing progress
docker logs whisperx-subtitle-translator | grep -E "(Processing|Translating|✅)"
```

## 🛠️ Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Verify container logs
docker logs whisperx-subtitle-translator

# Check health status
docker inspect whisperx-subtitle-translator | grep '"Status"'
```

#### GPU Not Detected
- Check NVIDIA driver version: `nvidia-smi`
- Verify Docker NVIDIA runtime: `docker info | grep nvidia`
- Restart Docker: `sudo systemctl restart docker`

#### Out of Memory
- Reduce `BATCH_SIZE` from 8 to 4 or 2
- Use smaller model: `medium` instead of `large-v3`
- Enable `TF32_ENABLED=1` for lower precision
- Check memory limits: increase container memory

#### Translation Errors
- Check internet connection (Hugging Face model downloads)
- Verify language code is supported
- Check `/config/cache` directory permissions
- Increase `TRANSLATION_TIMEOUT` value

#### Slow Processing
- Ensure GPU is being used: Check logs for "Using GPU"
- Increase `BATCH_SIZE` if you have sufficient VRAM
- Set `GPU_PRESET` to your specific GPU model
- Enable `TF32_ENABLED=1` for speed (with minimal accuracy loss)

### Log Analysis
```bash
# Check GPU usage
docker logs whisperx-subtitle-translator | grep "GPU"

# Monitor translation progress  
docker logs whisperx-subtitle-translator | grep "Translating"

# Check for errors
docker logs whisperx-subtitle-translator | grep "❌"

# View clean output only
docker logs whisperx-subtitle-translator | grep -v "UserWarning"
```

## 🔄 Updates

### Version History

#### v1.6.3 (Latest) - Clean Output Edition
- ✅ **Clean Log Output** - Suppressible library warnings (`SUPPRESS_LIBRARY_WARNINGS=1`)
- ✅ **Health Monitoring** - Container health checks with GPU verification
- ✅ **Memory Management** - 12GB RAM limit with 16GB swap
- ✅ **RTX 50 Series Presets** - GPU-specific optimization presets
- ✅ **Advanced Configuration** - Complete XML template with all settings
- ✅ **Performance Improvements** - Better batch processing efficiency

#### v1.6.2 (Stable)
- ✅ Fixed TF32 configuration issues
- ✅ Improved error handling and crash recovery
- ✅ Conservative defaults for maximum stability

#### v1.6.1
- ✅ Enhanced TF32 support for RTX 50 series
- ✅ PyTorch 2.9+ API compatibility
- ✅ Improved precision control

#### v1.6.0
- ✅ Optimized batch translation for GPU efficiency
- ✅ Enhanced Hugging Face model handling
- ✅ Better subtitle count verification
- ✅ Performance optimizations for RTX 50 series

#### v1.5
- ✅ Fixed text splitting issues during translation
- ✅ Improved timing preservation
- ✅ Added subtitle count verification

#### v1.4
- ✅ Maximum accuracy mode with float32 precision
- ✅ Word-level alignment support
- ✅ Enhanced timing verification

### Updating
```bash
# Pull latest version
docker pull agoddrie/whisperx-translator:latest

# Update using Docker Compose
docker-compose down && docker-compose pull && docker-compose up -d

# Manual container update
docker stop whisperx-subtitle-translator
docker rm whisperx-subtitle-translator
# Run new container with latest image
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
git clone https://github.com/AGo444/SRT.git
cd SRT
docker build -t whisperx-translator:dev .
```

### Testing
```bash
# Test specific GPU preset
docker run --gpus all -e GPU_PRESET=rtx5070ti whisperx-translator:dev

# Test with different languages
docker run --gpus all -e TARGET_LANGUAGE=de whisperx-translator:dev
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [WhisperX](https://github.com/m-bain/whisperX) - Enhanced Whisper with alignment
- [Hugging Face](https://huggingface.co/) - Translation models and transformers library
- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) - OPUS-MT translation models
- [NVIDIA](https://developer.nvidia.com/) - CUDA and cuDNN libraries

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/AGo444/SRT/issues)
- **Docker Hub**: [agoddrie/whisperx-translator](https://hub.docker.com/r/agoddrie/whisperx-translator)
- **Community**: [Unraid Community Forums](https://forums.unraid.net/)

## 🎯 Quick Start Examples

### RTX 5070 Ti Users (Recommended)
```bash
docker run -d \
  --name="whisperx-rtx5070ti" \
  --runtime=nvidia --gpus all \
  --memory=12g --memory-swap=16g \
  -e TARGET_LANGUAGE="nl" \
  -e GPU_PRESET="rtx5070ti" \
  -e WHISPERX_MODEL="large-v3" \
  -e DEBUG="0" \
  -e SUPPRESS_LIBRARY_WARNINGS="1" \
  -v "/path/to/videos:/data:rw" \
  -v "/path/to/cache:/config/cache:rw" \
  agoddrie/whisperx-translator:1.6.3
```

### RTX 4060 Users (Memory Optimized)
```bash
docker run -d \
  --name="whisperx-rtx4060" \
  --runtime=nvidia --gpus all \
  --memory=8g --memory-swap=12g \
  -e TARGET_LANGUAGE="nl" \
  -e GPU_PRESET="rtx4060" \
  -e WHISPERX_MODEL="medium" \
  -e BATCH_SIZE="2" \
  -e COMPUTE_TYPE="float16" \
  -v "/path/to/videos:/data:rw" \
  -v "/path/to/cache:/config/cache:rw" \
  agoddrie/whisperx-translator:1.6.3
```

---

**Made with ❤️ for the Unraid Community | Optimized for RTX 50 Series GPUs**

*Latest update: v1.6.3 - The Clean Output Edition*