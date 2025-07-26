import os
import subprocess
import pysrt
import sys
import argparse
import re
import torch
import gc
import logging
import time  # Voeg deze import toe aan het begin van het bestand

# --- Configuration ---
DEFAULT_TARGET_LANGUAGE = os.getenv('TARGET_LANGUAGE', 'nl')
WHISPERX_MODEL = os.getenv('WHISPERX_MODEL', 'large-v3')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '8'))
DEBUG = os.getenv('DEBUG', '0') == '1'
USE_CUDA = os.getenv('USE_CUDA', '1') == '1'
DEVICE = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"

# Make INPUT_DIR configurable
INPUT_DIR = os.getenv('INPUT_DIR', '/data')
HF_TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-en-nl"
SUPPORTED_LANGUAGES = ['nl', 'de', 'fr', 'es', 'it']  # Add all supported languages

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def clean_filename(filename):
    """
    Removes potentially problematic characters from a filename,
    especially useful for paths in shell commands.
    """
    return re.sub(r'[^\w\s\-\.\_]', '', filename).strip()

def is_video_file(filepath):
    """Checks if a file is a common video format."""
    return filepath.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv'))

def get_target_language_prefix(target_language):
    """Returns the correct prefix for the Hugging Face translation model."""
    return f">>{target_language}<<"

def split_long_text(text, max_length=100):
    """Split text into smaller chunks at punctuation marks."""
    if len(text) <= max_length:
        return [text]
    
    # Try to split at these punctuation marks
    split_chars = ['. ', '! ', '? ', '; ', ': ', ', ']
    
    for split_char in split_chars:
        if split_char in text:
            parts = text.split(split_char)
            result = []
            current_part = ""
            
            for part in parts:
                if len(current_part + part + split_char) > max_length and current_part:
                    result.append(current_part.strip())
                    current_part = part + split_char
                else:
                    current_part += part + split_char
            
            if current_part:
                result.append(current_part.strip())
            return result
    
    # If no punctuation found, split at max_length
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Update de generate_subtitles functie
def generate_subtitles(video_path, output_dir, target_language):
    logger.debug(f"Starting subtitle generation for {video_path}")
    logger.debug(f"Output directory: {output_dir}")
    logger.debug(f"Target language: {target_language}")

    """
    Generates English subtitles using WhisperX and then translates them.cccccbhlfljdbcctlufkfkcjkbjgblkiicrndiujcb
    Handles existing SRT files to resume processing.
    """
    # Add validation at the start of the function
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file '{video_path}' does not exist.")
        return
    
    if not os.path.exists(output_dir):
        print(f"❌ Error: Output directory '{output_dir}' does not exist.")
        return

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_base_path = os.path.join(output_dir, base_name)
    english_srt_path = f"{output_base_path}.en.srt"
    target_srt_path = f"{output_base_path}.{target_language}.srt"

    # --- Logging for ALL existing SRT files related to this video ---
    found_any_srt = False
    for filename in os.listdir(output_dir):
        if filename.startswith(base_name) and filename.endswith('.srt'):
            full_srt_path = os.path.join(output_dir, filename)
            if os.path.isfile(full_srt_path) and os.path.getsize(full_srt_path) > 0:
                print(f"Found existing SRT: {filename}")
                found_any_srt = True
    if not found_any_srt:
        print("Found no existing .srt files for this video.")
    # --- End logging for ALL existing SRT files ---

    # Check if target language SRT already exists and is not empty
    if os.path.exists(target_srt_path) and os.path.getsize(target_srt_path) > 0:
        print(f"✅ Skipping '{video_path}': '{target_language}' subtitles already exist and are not empty.")
        return

    # Check if English SRT exists and is not empty
    if os.path.exists(english_srt_path) and os.path.getsize(english_srt_path) > 0:
        print(f"➡️ English subtitles already exist for '{video_path}'. Proceeding to translate.")
    else:
        print(f"🎧 Generating EN subtitles for: {video_path}")
        whisperx_command = [
            "whisperx",
            video_path,
            "--model", WHISPERX_MODEL,
            "--output_dir", output_dir,
            "--output_format", "srt",
            "--language", "en",
            "--batch_size", "1" if DEVICE == "cpu" else "8",
            "--compute_type", "int8" if DEVICE == "cpu" else "float32",
            "--device", DEVICE,
            "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H"
        ]
        
        start_time = time.time()
        print(f"⏱️ Started processing at {time.strftime('%H:%M:%S')} using {DEVICE.upper()}")
        
        try:
            process = subprocess.Popen(
                whisperx_command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1, 
                universal_newlines=True
            )

            # Add timeout check
            max_silence = 300  # 5 minutes
            last_output_time = time.time()

            for line in iter(process.stdout.readline, ''):
                current_time = time.time()
                sys.stdout.write(line)
                sys.stdout.flush()

                # Reset timer on any output
                if line.strip():
                    last_output_time = current_time

                # Check for timeout
                if current_time - last_output_time > max_silence:
                    process.kill()
                    print(f"\n❌ Process appears to be stuck (no output for {max_silence} seconds). Killing process...")
                    return

                if "Processing:" in line or "Performing alignment" in line:
                    elapsed = current_time - start_time
                    print(f"⏳ Processing time so far: {int(elapsed//60)}m {int(elapsed%60)}s")
                
                # Force garbage collection during long processes
                if "Performing alignment" in line:
                    gc.collect()
                    torch.cuda.empty_cache() if DEVICE == "cuda" else None

            process.wait(timeout=3600)  # 1 hour timeout
            if process.returncode == 0:
                # Check if WhisperX created the file with a different name
                potential_files = [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith('.srt')]
                for file in potential_files:
                    if '.en.' not in file:
                        # Rename to include language code if needed
                        old_path = os.path.join(output_dir, file)
                        new_path = english_srt_path
                        if os.path.exists(old_path):
                            os.rename(old_path, new_path)
                            print(f"Renamed {file} to {os.path.basename(english_srt_path)}")
            else:
                print(f"❌ WhisperX processing failed with return code {process.returncode}")

            total_time = time.time() - start_time
            print(f"✅ Processing completed in {int(total_time//60)}m {int(total_time%60)}s")
            # --- Log that English file is created ---
            print(f"Created {os.path.basename(english_srt_path)}")
            # --- End log ---

        except subprocess.CalledProcessError as e:
            print(f"❌ Error during WhisperX processing for '{video_path}':")
            print(f"Command: {e.cmd}")
            print(f"Return Code: {e.returncode}")
            print(f"Output: {e.output}")
            return
        except FileNotFoundError:
            print(f"❌ Error: 'whisperx' command not found. Is WhisperX installed correctly in the container?")
            return

    # Translate the English SRT to the target language
    if os.path.exists(english_srt_path) and os.path.getsize(english_srt_path) > 0:
        print(f"🌐 Translating '{english_srt_path}' to '{target_language}' with Hugging Face model '{HF_TRANSLATE_MODEL}'...")
        try:
            translate_srt(english_srt_path, target_srt_path, target_language)
            print(f"✅ Translated subtitles saved as: {target_srt_path}")
        except Exception as e:
            print(f"❌ Error during translation for '{video_path}': {e}")
    else:
        print(f"⚠️ No English SRT found for '{video_path}' after transcription. Skipping translation.")

    print(f"\n--- Summary for {os.path.basename(video_path)} ---")
    print(f"English SRT: {english_srt_path} {'(Generated)' if os.path.exists(english_srt_path) and os.path.getsize(english_srt_path) > 0 and not os.path.exists(target_srt_path) else '(Exists)' if os.path.exists(english_srt_path) and os.path.getsize(english_srt_path) > 0 else '(Failed/Missing)'}")
    print(f"Target ({target_language}) SRT: {target_srt_path} {'(Generated)' if os.path.exists(target_srt_path) and os.path.getsize(target_srt_path) > 0 else '(Failed/Missing)'}")
    print("------------------------------------------\n")


# Voeg dynamische model selectie toe
def get_translation_model(target_language):
    language_model_map = {
        'nl': 'Helsinki-NLP/opus-mt-en-nl',
        'de': 'Helsinki-NLP/opus-mt-en-de',
        'fr': 'Helsinki-NLP/opus-mt-en-fr',
        'es': 'Helsinki-NLP/opus-mt-en-es',
        'it': 'Helsinki-NLP/opus-mt-en-it'
    }
    return language_model_map.get(target_language)

# Update de translate_srt functie
def translate_srt(input_srt_path, output_srt_path, target_language):
    model_name = get_translation_model(target_language)
    if not model_name:
        raise ValueError(f"Unsupported target language: {target_language}")
    
    try:
        from transformers import pipeline
        translator = pipeline("translation", model=model_name, device=-1 if DEVICE == "cpu" else 0)
        print(f"✅ Hugging Face translation model loaded on {DEVICE}")
        
        # Add test translation to verify model
        test_result = translator(">>nl<< This is a test sentence to check if model is working.")
        print(f"🔍 Model test translation: {test_result[0]['translation_text']}")
        
    except Exception as e:
        print(f"❌ Error loading translation model: {e}")
        raise

    try:
        subs = pysrt.open(input_srt_path, encoding='utf-8')
    except Exception as e:
        print(f"❌ Error reading source SRT file: {e}")
        raise

    translated_subs = pysrt.SubRipFile()
    translated_texts = {}  # Initialize dictionary for translations

    lang_prefix = get_target_language_prefix(target_language)
    texts_to_translate = [f"{lang_prefix} {sub.text}" for sub in subs]
    
    # Kleinere batch size voor CPU
    batch_size = 1 if DEVICE == "cpu" else 16
    for i in range(0, len(texts_to_translate), batch_size):
        batch = texts_to_translate[i:i+batch_size]
        
        # Split long sentences
        split_batch = []
        for text in batch:
            if len(text) > 100:  # Split if longer than 100 characters
                split_batch.extend(split_long_text(text))
            else:
                split_batch.append(text)
        
        translations = translator(split_batch)
        
        # Combine translations if needed
        for j, translation in enumerate(translations):
            translated_text = translation['translation_text']
            if len(translated_text) > 100:
                # Add line breaks for readability
                translated_text = translated_text.replace('. ', '.\n')
                translated_text = translated_text.replace('! ', '!\n')
                translated_text = translated_text.replace('? ', '?\n')
            
            translated_texts[i+j] = translated_text
    
    # Create new subtitles with translations
    for index, sub in enumerate(subs):
        new_sub = pysrt.SubRipItem(
            index=sub.index,
            start=sub.start,
            end=sub.end,
            text=translated_texts[index]
        )
        translated_subs.append(new_sub)
    
    translated_subs.save(output_srt_path, encoding='utf-8')

# --- Device Configuration ---
def get_device():
    if torch.cuda.is_available() and os.getenv('USE_CUDA', '1') == '1':
        print("✅ Using GPU for processing")
        return "cuda"
    print("⚠️ Using CPU for processing - this will be slower!")
    return "cpu"

# Device globally beschikbaar maken
DEVICE = get_device()

# --- Main Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and translate subtitles for video files.")
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_TARGET_LANGUAGE,
        help=f"Target language for translation (supported: {', '.join(SUPPORTED_LANGUAGES)}). Defaults to '{DEFAULT_TARGET_LANGUAGE}'."
    )
    args = parser.parse_args()

    target_language = args.language.lower()
    if target_language not in SUPPORTED_LANGUAGES:
        print(f"❌ Error: Unsupported language '{target_language}'. Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
        sys.exit(1)

    print(f"Starting subtitle generation and translation process for target language: {target_language}")

    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Input directory '{INPUT_DIR}' does not exist. Please ensure your volume is mounted correctly.")
        sys.exit(1)

    found_videos = False
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if is_video_file(file):
                found_videos = True
                video_full_path = os.path.join(root, file)
                print(f"\n--- Processing video: {video_full_path} ---")
                generate_subtitles(video_full_path, root, target_language)

    if not found_videos:
        print(f"No supported video files (.mp4, .mkv, etc.) found in '{INPUT_DIR}' or its subdirectories.")
        print("Please ensure your video files are in the mounted /data directory.")