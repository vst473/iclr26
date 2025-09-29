#!/usr/bin/env python3
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
from pathlib import Path
import argparse

error_files = set()


def load_embeddings_from_jsonl(filename):
    """
    Load embeddings and IDs from a JSONL file with error handling
    """
    embeddings = []
    ids = []
    skipped_count = 0
    total_lines = 0

    print(f"📖 Reading {filename}...")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                try:
                    json_obj = json.loads(line.strip())

                    if 'embedding' not in json_obj:
                        print(f"⚠️  Line {line_num}: Missing 'embedding' key")
                        skipped_count += 1
                        continue

                    if not json_obj['embedding'] or len(json_obj['embedding']) == 0:
                        print(f"⚠️  Line {line_num}: Empty embedding")
                        skipped_count += 1
                        continue

                    embedding_vector = json_obj['embedding'][0]
                    if not embedding_vector or len(embedding_vector) == 0:
                        print(f"⚠️  Line {line_num}: Invalid embedding vector")
                        skipped_count += 1
                        continue

                    embeddings.append(embedding_vector)
                    ids.append(json_obj.get('id', f'missing_id_{line_num}'))

                except json.JSONDecodeError as e:
                    print(f"❌ Line {line_num}: JSON decode error - {e}")
                    skipped_count += 1
                    continue
                except Exception as e:
                    print(f"❌ Line {line_num}: Unexpected error - {e}")
                    skipped_count += 1
                    continue
    except Exception as e:
        print(f"Failed to open {filename}: {e}")
        error_files.add(str(filename))
        return [], np.array([])

    if skipped_count > 0:
        print(f"Skipped {skipped_count}/{total_lines} lines due to errors")

    if not embeddings:
        print(f"No valid embeddings found in {filename}")
        error_files.add(str(filename))
        return [], np.array([])

    print(f"Loaded {len(embeddings)} valid embeddings from {filename}")
    return ids, np.array(embeddings)


def calculate_similarities_by_id(eng_ids, eng_embeddings, trans_ids, trans_embeddings, filename):
    """
    Calculate cosine similarity by matching IDs
    """
    if len(trans_embeddings) == 0:
        error_files.add(str(filename))
        return [], [], eng_ids

    trans_dict = {tid: trans_embeddings[i] for i, tid in enumerate(trans_ids) if tid}

    similarities = []
    matched_ids = []
    unmatched_ids = []

    for i, eng_id in enumerate(eng_ids):
        if eng_id and eng_id in trans_dict:
            try:
                eng_emb = eng_embeddings[i].reshape(1, -1)
                trans_emb = trans_dict[eng_id].reshape(1, -1)

                similarity = cosine_similarity(eng_emb, trans_emb)[0][0]
                similarities.append(similarity)
                matched_ids.append(eng_id)
            except Exception as e:
                print(f"Error calculating similarity for ID {eng_id}: {e}")
                unmatched_ids.append(eng_id)
                error_files.add(str(filename))
        else:
            unmatched_ids.append(eng_id)

    return matched_ids, similarities, unmatched_ids


def process_all_languages(english_file, translated_files):
    print("Loading English embeddings...")
    eng_ids, eng_embeddings = load_embeddings_from_jsonl(english_file)

    if len(eng_embeddings) == 0:
        print("No English embeddings loaded. Exiting.")
        error_files.add(str(english_file))
        return {}, []

    results = {}

    for trans_file in translated_files:
        if not os.path.exists(trans_file):
            print(f"File not found: {trans_file}")
            error_files.add(str(trans_file))
            continue

        print(f"\nProcessing: {trans_file}")
        lang_name = Path(trans_file).stem

        try:
            trans_ids, trans_embeddings = load_embeddings_from_jsonl(trans_file)

            if len(trans_embeddings) == 0:
                print(f"No valid embeddings in {lang_name}")
                error_files.add(str(trans_file))
                continue

            matched_ids, similarities, unmatched_ids = calculate_similarities_by_id(
                eng_ids, eng_embeddings, trans_ids, trans_embeddings, trans_file
            )

            if not similarities:
                print(f"No matching IDs found for {lang_name}")
                error_files.add(str(trans_file))
                continue

            results[lang_name] = {
                'matched_ids': matched_ids,
                'similarities': similarities,
                'unmatched_ids': unmatched_ids,
                'total_matches': len(matched_ids),
                'total_unmatched': len(unmatched_ids),
                'match_percentage': (len(matched_ids) / len(eng_ids)) * 100,
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'min_similarity': np.min(similarities),
                'max_similarity': np.max(similarities)
            }

            print(f"Matched {len(matched_ids)}/{len(eng_ids)} IDs "
                  f"({results[lang_name]['match_percentage']:.1f}%)")
            print(f"Mean cosine similarity: {results[lang_name]['mean_similarity']:.4f}")

        except Exception as e:
            print(f"Error processing {trans_file}: {str(e)}")
            error_files.add(str(trans_file))
            continue

    return results, eng_ids


def save_detailed_results(results, eng_ids, output_file):
    if not results:
        print("No results to save")
        return

    all_data = []
    all_matched_ids = set(eng_ids)
    for lang_results in results.values():
        all_matched_ids.update(lang_results['matched_ids'])

    for id_val in sorted(all_matched_ids):
        row = {'id': id_val}
        for lang_name, lang_results in results.items():
            if id_val in lang_results['matched_ids']:
                idx = lang_results['matched_ids'].index(id_val)
                row[f'{lang_name}_similarity'] = lang_results['similarities'][idx]
            else:
                row[f'{lang_name}_similarity'] = None
        all_data.append(row)

    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")


def save_summary_results(results, output_file):
    if not results:
        print("No results to save")
        return

    summary_data = []
    for lang_name, lang_results in results.items():
        summary_data.append({
            'Language': lang_name,
            'Total_Matches': lang_results['total_matches'],
            'Total_Unmatched': lang_results['total_unmatched'],
            'Match_Percentage': round(lang_results['match_percentage'], 2),
            'Mean_Similarity': round(lang_results['mean_similarity'], 4),
            'Std_Similarity': round(lang_results['std_similarity'], 4),
            'Min_Similarity': round(lang_results['min_similarity'], 4),
            'Max_Similarity': round(lang_results['max_similarity'], 4)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Mean_Similarity', ascending=False)
    summary_df.to_csv(output_file, index=False)
    print(f"Summary results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="ID-based cosine similarity analysis")
    parser.add_argument("--input_dir", required=True, help="Folder with JSONL files (English + translations)")
    parser.add_argument("--output_dir", required=True, help="Folder to save CSV results")
    parser.add_argument("--english_file", default="mmlu_qwen3_embd_1.jsonl", help="English embeddings JSONL filename")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    english_file = input_dir / args.english_file
    translated_files = [f for f in input_dir.glob("*.jsonl") if f.name != args.english_file]

    print("Starting ID-based cosine similarity analysis...")
    print(f"English file: {english_file}")
    print(f"Translated files: {len(translated_files)} languages")

    results, eng_ids = process_all_languages(english_file, translated_files)

    if results:
        save_detailed_results(results, eng_ids, output_dir / "similarity_results_detailed.csv")
        save_summary_results(results, output_dir / "similarity_summary.csv")
    else:
        print("No results to display.")

    # Print all error files at the end
    print("\n" + "=" * 60)
    if error_files:
        print("The following files had errors:")
        for ef in sorted(error_files):
            print(f"   - {ef}")
    else:
        print("No errors encountered in any file")
    print("=" * 60)


if __name__ == "__main__":
    main()
