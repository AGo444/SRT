# Stage 1: GPU-accelerated ffmpeg met NVENC/NVDEC
FROM jrottenberg/ffmpeg:6.0-nvidia AS ffmpeg-nv

# Stage 2: PyTorch + WhisperX omgeving
FROM nvcr.io/nvidia/pytorch:25.04-py3
LABEL com.nvidia.volumes.needed="nvidia_driver"

# Werkdirectory
WORKDIR /app

# Kopieer ffmpeg/ffprobe met GPU-acceleratie
COPY --from=ffmpeg-nv /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg-nv /usr/local/bin/ffprobe /usr/local/bin/ffprobe


# Installeer systeemafhankelijkheden: ffmpeg voor video, git voor kloon, python3-pip, libcudnn voor GPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    python3-pip \
    libcudnn8 libcudnn8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installeer eerst requirements.txt voor betere cache-benutting
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Kopieer applicatie bestanden als laatste
COPY createSrt.py .

# Geef aan dat /data de plek is waar de video's komen
VOLUME /data

# Zorg ervoor dat de uitvoerbuffer van Python direct wordt geflusht
ENV PYTHONUNBUFFERED=1

# Maak een non-root gebruiker aan
RUN useradd -m -s /bin/bash appuser
USER appuser

# Zorg dat /data schrijfbaar is voor appuser
RUN mkdir -p /data && chown appuser:appuser /data

# Expliciete CUDA configuratie
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,video

# Voeg expliciete error handling toe voor GPU checks
RUN python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA niet beschikbaar!'"

# Voeg environment variabelen toe voor betere taal ondersteuning
ENV LANGUAGE_MODELS="Helsinki-NLP/opus-mt-en-nl Helsinki-NLP/opus-mt-en-de Helsinki-NLP/opus-mt-en-fr Helsinki-NLP/opus-mt-en-es Helsinki-NLP/opus-mt-en-it"

# Predownload modellen voor offline gebruik
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration; \
    for model in '$LANGUAGE_MODELS'.split(): \
        AutoTokenizer.from_pretrained(model); \
        AutoModelForSeq2SeqGeneration.from_pretrained(model)"

# Definieer de entrypoint van de container. Argumenten die aan 'docker run' worden meegegeven,
# worden hierachter geplakt.
ENTRYPOINT ["python3", "createSrt.py"]

# Standaard commando als er geen argumenten worden meegegeven.
# Dit zal de standaardtaal "nl" gebruiken als argument voor het script.
CMD ["--language", "nl"]

# Voeg healthcheck toe
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('GPU available:', torch.cuda.is_available())" || exit 1