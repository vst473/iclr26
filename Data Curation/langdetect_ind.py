import fasttext
import os
import sys
from tqdm import tqdm
import argparse

LANGUAGE_MAP = {
    "hindi": "hi",
    "devanagari": "hi", 
    "bengali": "bn",
    "assamese": "as",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "oriya": "or",
    "odia": "or",  
    "punjabi": "pa",
    "gurmukhi": "pa", 
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur",
    "tibetan": "bo",
    "limbu": "lif" 
}


SCRIPT_MAP = {
    "hindi": "Devanagari",
    "devanagari": "Devanagari",
    "bengali": "Bengali",
    "assamese": "Bengali",
    "gujarati": "Gujarati",
    "kannada": "Kannada",
    "malayalam": "Malayalam",
    "oriya": "Oriya",
    "odia": "Oriya",
    "punjabi": "Gurmukhi",
    "gurmukhi": "Gurmukhi",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "urdu": "Arabic",
    "tibetan": "Tibetan",
    "limbu": "Limbu"
}

def count_lines_in_file(file_path):
    """Count the number of lines in a given file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            return sum(1 for _ in file)
    except Exception as e:
        print(f"Error counting lines in {file_path}: {e}")
        return 0

def load_fasttext_model(model_path):
    """Load the FastText model from the specified path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading FastText model from {model_path}...")
    try:
        model = fasttext.load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def detect_language(model, text):

    if not text or text.isspace():
        return "unknown"
        
    try:
        
        labels, scores = model.predict(text.strip(), k=1)
        lang_code = labels[0].replace("__label__", "") if labels else "unknown"
        confidence = scores[0] if scores else 0
        
        
        if confidence > 0.5:
            return lang_code
        return "unknown"
    except Exception:
        return "unknown"

def contains_script_characters(text, language):
    lang_lower = language.lower()
    
    script_ranges = {
        "Devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF), (0x1CD0, 0x1CFF)),
        "Bengali": ((0x0980, 0x09FF),),
        "Gujarati": ((0x0A80, 0x0AFF),),
        "Gurmukhi": ((0x0A00, 0x0A7F),),
        "Kannada": ((0x0C80, 0x0CFF),),
        "Malayalam": ((0x0D00, 0x0D7F),),
        "Oriya": ((0x0B00, 0x0B7F),),
        "Tamil": ((0x0B80, 0x0BFF),),
        "Telugu": ((0x0C00, 0x0C7F),),
        "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)),
        "Tibetan": ((0x0F00, 0x0FFF),),
        "Limbu": ((0x1900, 0x194F),)
    }
    
    script = SCRIPT_MAP.get(lang_lower)
    if not script or script not in script_ranges:
        return True 
    
    ranges = script_ranges[script]
    for char in text:
        code = ord(char)
        for start, end in ranges:
            if start <= code <= end:
                return True
    
    return False

def extract_language_lines(input_file, output_file, model_path, languages, min_chars=5, strict_mode=False):
    model = load_fasttext_model(model_path)
    
    lang_codes = {}
    for lang in languages:
        lang_lower = lang.lower()
        if lang_lower in LANGUAGE_MAP:
            lang_codes[LANGUAGE_MAP[lang_lower]] = lang
        else:
            print(f"Warning: Language '{lang}' not recognized or supported by FastText")
    
    if not lang_codes:
        print("Error: No valid languages specified")
        return
    
    output_files = {}
    for code, lang in lang_codes.items():
        if "{lang}" in output_file:
            out_path = output_file.format(lang=lang.lower())
        else:
            base, ext = os.path.splitext(output_file)
            out_path = f"{base}_{lang.lower()}{ext}"
        output_files[code] = out_path
    
    os.makedirs(os.path.dirname(os.path.abspath(list(output_files.values())[0])), exist_ok=True)
    
    language_counts = {code: 0 for code in lang_codes}
    language_buffers = {code: [] for code in lang_codes}
    buffer_size = 1000 
    
    total_lines = count_lines_in_file(input_file)
    lines_processed = 0
    
    print(f"Processing {total_lines} lines from {input_file}")
    
    with open(input_file, "r", encoding="utf-8", errors="ignore") as infile:
        for line in tqdm(infile, desc="Processing lines", total=total_lines):
            line = line.strip()
            lines_processed += 1
            
            if len(line) < min_chars:
                continue
                
            lang_code = detect_language(model, line)
            
            if lang_code in lang_codes:
                if strict_mode and not contains_script_characters(line, lang_codes[lang_code]):
                    continue
                
                language_buffers[lang_code].append(line)
                language_counts[lang_code] += 1
                
                if len(language_buffers[lang_code]) >= buffer_size:
                    with open(output_files[lang_code], "a", encoding="utf-8") as outfile:
                        outfile.write("\n".join(language_buffers[lang_code]) + "\n")
                    language_buffers[lang_code] = []
    
    for lang_code, buffer in language_buffers.items():
        if buffer:
            with open(output_files[lang_code], "a", encoding="utf-8") as outfile:
                outfile.write("\n".join(buffer) + "\n")
    
    print("\nDetection completed:")
    print(f"Total lines processed: {lines_processed}")
    
    for lang_code, count in language_counts.items():
        lang_name = lang_codes[lang_code]
        percent = (count / lines_processed) * 100 if lines_processed > 0 else 0
        print(f"{lang_name} ({lang_code}): {count} lines ({percent:.2f}%)")
        if count > 0:
            print(f"  Saved to: {output_files[lang_code]}")

def get_available_languages():
    """Return a formatted string of available languages with their codes."""
    languages = []
    for lang, code in sorted(LANGUAGE_MAP.items()):
        if code not in [item[1] for item in languages]:
            languages.append((lang, code))
    
    result = "Available languages:\n"
    for lang, code in languages:
        result += f"  {lang} ({code})\n"
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract lines in specific languages from a text file using FastText.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=get_available_languages()
    )
    
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Path to the input text file")
    parser.add_argument("--output", "-o", type=str, default="./output_{lang}.txt", 
                        help="Output file path template. Use {lang} for language name insertion")
    parser.add_argument("--model", "-m", type=str, default="fasttext.bin", 
                        help="Path to the FastText model (e.g., lid.176.bin)")
    parser.add_argument("--languages", "-l", type=str, nargs='+', required=True,
                        help="List of languages to detect (e.g., hindi bengali tamil)")
    parser.add_argument("--min-chars", type=int, default=5,
                        help="Minimum characters for a valid line")
    parser.add_argument("--strict", action="store_true",
                        help="Enable strict mode with additional script verification")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Please download a FastText model (e.g., lid.176.bin) from https://fasttext.cc/docs/en/language-identification.html")
        sys.exit(1)
    
    extract_language_lines(
        args.input, 
        args.output, 
        args.model, 
        args.languages, 
        args.min_chars,
        args.strict
    )
