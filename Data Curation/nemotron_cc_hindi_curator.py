import argparse
import os
import shutil
from typing import Any, List
import time
import json
import argparse

from modifiers import HTMLTagModifier, QuotationUnifier

from nemo_curator import ScoreFilter, Sequential, AddId, FuzzyDuplicatesConfig, TaskDecontamination
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import (
    RepeatingTopNGramsFilter,
    WordCountFilter, 
    SymbolsToWordsFilter,
    NumbersFilter,
    UrlsFilter,
    WhiteSpaceFilter,
    RepeatingDuplicateNGramsFilter,
    RepeatedParagraphsFilter,
    NonAlphaNumericFilter,
    ParenthesesFilter,
    RepeatedLinesFilter,
    MeanWordLengthFilter
)
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.modifiers import BoilerPlateStringModifier
from nemo_curator.modules import ExactDuplicates
from nemo_curator import FuzzyDuplicates, FuzzyDuplicatesConfig
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import get_all_files_paths_under
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from nemo_curator.classifiers import QualityClassifier

from nemo_curator.tasks import (
    ANLI, CB, PIQA, RTE, WSC, ArcChallenge, ArcEasy,
    BoolQ, Copa, Drop, MultiRC, OpenBookQA, Quac,
    Race, Record, Squad, TriviaQA, WebQA, WiC, Winogrande,
)
from hindi_tasks import (
    HindiCSQA, HindiSA, HindiWSTP, HindiNER, HindiSQuAD,
    HindiXNLI, HindiPOS, HindiWanli, HindiMnli, HindiFever
 )
import fasttext

model_path = "model_fasttext.bin"
# os.environ['DASK_DATAFRAME__QUERY_PLANNING'] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEXT_FIELD = "text"

def pre_imports():
    import cudf

def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' removed successfully.")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except PermissionError:
        print(f"Permission denied: '{file_path}'.")
    except Exception as e:
        print(f"Error occurred while trying to remove the file: {e}")


def clean_and_unify(dataset: DocumentDataset) -> DocumentDataset:
    start = time.time()
    cleaners = Sequential(
        [
            Modify(BoilerPlateStringModifier(), text_field=TEXT_FIELD),
            Modify(HTMLTagModifier(), text_field=TEXT_FIELD),
            Modify(UnicodeReformatter(), text_field=TEXT_FIELD),
            Modify(QuotationUnifier(), text_field=TEXT_FIELD),
        ]
    )
    cleaned_dataset = cleaners(dataset)
    print(f"\nclean_and_unify completed. Removed {len(dataset.df)-len(cleaned_dataset.df)} rows")
    print(f"Time taken {time.time()-start}s")
    return cleaned_dataset

# def quality_filter(dataset: DocumentDataset) -> DocumentDataset:
#     start = time.time()
#     quality_classifier = QualityClassifier(filter_by=["High", "Medium"])
#     filtered_dataset = quality_classifier(dataset)
#     print(f"\nquality_filter completed. Removed {len(dataset.df)-len(filtered_dataset.df)} rows")
#     print(f"Time taken {time.time()-start}s")
#     return filtered_dataset


def heuristic_filter(dataset: DocumentDataset) -> DocumentDataset:
    start = time.time()
    filters = Sequential(
        [
            ScoreFilter(WordCountFilter(min_words=50, max_words=10000),text_field=TEXT_FIELD,score_field="word_count",score_type=int),
            ScoreFilter(RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2), text_field=TEXT_FIELD, score_type=float),
            ScoreFilter(RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18), text_field=TEXT_FIELD, score_type=float),
            ScoreFilter(SymbolsToWordsFilter(max_symbol_to_word_ratio=0.1), text_field=TEXT_FIELD, score_type=float),
            ScoreFilter(NumbersFilter(max_number_to_text_ratio=0.15), text_field=TEXT_FIELD, score_type=float),
            ScoreFilter(UrlsFilter(max_url_to_text_ratio=0.2), text_field=TEXT_FIELD, score_type=float),
            ScoreFilter(WhiteSpaceFilter(max_white_space_ratio=0.25), text_field=TEXT_FIELD, score_type=float),
            # ScoreFilter(NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25), text_field=TEXT_FIELD, score_type=float),
        ]
    )
    filtered_dataset = filters(dataset)
    print(f"\nheuristic_filter completed. Removed {len(dataset.df)-len(filtered_dataset.df)} rows")
    print(f"Time taken {time.time()-start}s")
    return filtered_dataset

def dedupe(dataset: DocumentDataset) -> DocumentDataset:
    start = time.time()

    ######################## Exact dedup ########################
    add_id = AddId(id_field='id',id_prefix='mix_data',start_index=0)
    dataset = add_id(dataset)
    
    #deduplicator = ExactDuplicates(id_field="id", text_field="raw_content", hash_method="md5")
    deduplicator = ExactDuplicates(id_field="id", text_field=TEXT_FIELD, hash_method="md5")

    duplicates = deduplicator(dataset)
    docs_to_remove = duplicates.df.map_partitions(
        lambda x: x[x._hashes.duplicated(keep="first")]
    )

    duplicate_ids = list(docs_to_remove.compute().id)
    dataset_df = dataset.df
    deduped = dataset_df[~dataset_df.id.isin(duplicate_ids)]
    deduped_dataset = DocumentDataset(deduped)
    print(f"\ndedupe completed. Removed {len(dataset.df)-len(deduped_dataset.df)} rows")
    print(f"Time taken {time.time()-start}s")
    return deduped_dataset

def redact_pii(dataset: DocumentDataset) -> DocumentDataset:
    start = time.time()
    redactor = Modify(
        PiiModifier(
            supported_entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD"],
            anonymize_action="replace",
            device="gpu",
        ),
        text_field=TEXT_FIELD
    )
    redacted_dataset = redactor(dataset)
    print(f"\nredact_pii completed. Removed {len(dataset.df)-len(redacted_dataset.df)} rows")
    print(f"Time taken {time.time()-start}s")
    return redacted_dataset

def create_predict_quality_function(model_path: str):
    """Creates a prediction function that can be properly serialized"""
    def predict_batch(texts: List[str]) -> List[str]:
        model = fasttext.load_model(model_path)
        results = []
        for text in texts:
            predicted_label, _ = model.predict(text, k=1)
            results.append(predicted_label[0].replace("__label__", ""))
        return results
    return predict_batch

def quality_filter(dataset: DocumentDataset) -> DocumentDataset:
    """
    Filter dataset based on quality predictions from FastText model.
    Processes data in batches to handle large datasets efficiently.
    """
    start = time.time()
    df = dataset.df.compute()
    batch_size = 1000
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    all_predictions = []
    predict_quality = create_predict_quality_function(model_path)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch_texts = df[TEXT_FIELD].iloc[start_idx:end_idx].tolist()
        batch_predictions = predict_quality(batch_texts)
        all_predictions.extend(batch_predictions)
    df['quality'] = all_predictions
    filtered_df = df[df['quality'].isin(["High", "Medium"])]
    filtered_dataset = DocumentDataset(dd.from_pandas(filtered_df, npartitions=dataset.df.npartitions))
    
    print(f"\nquality_filter completed. Removed {len(df) - len(filtered_df)} rows")
    print(f"Time taken: {time.time() - start}s")
    
    return filtered_dataset

def decontaminate(dataset: DocumentDataset) -> DocumentDataset:
    start = time.time()
    downstream_tasks = [
        ### ENGLISH BENCHMARKS ###
        Winogrande(), Squad(), TriviaQA(), Quac(), WebQA(),
        Race(), Drop(), WiC(), PIQA(), ArcEasy(), ArcChallenge(),
        OpenBookQA(), BoolQ(), Copa(), RTE(), MultiRC(), WSC(),
        CB(), ANLI(), Record(),
        ### HINDI BENCHMARKS ###
        HindiCSQA(), HindiSA(), HindiWSTP(), HindiNER(), HindiSQuAD(),
        HindiXNLI(), HindiPOS(), HindiWanli(), HindiMnli(), HindiFever()
    ]

    decontaminator = TaskDecontamination(downstream_tasks, text_field=TEXT_FIELD)
    decontaminated_dataset = decontaminator(dataset)
    print(f"\ndecontaminate completed. Removed {len(dataset.df)-len(decontaminated_dataset.df)} rows")
    print(f"Time taken {time.time()-start}s")
    return decontaminated_dataset

def process_chunk(file_path: str, file_name_inp: str) -> None:
    files = [file_path]
    print(f"Running curation pipeline on '{files[0]}'...")

    print("Reading the data...")
    orig_dataset = DocumentDataset.read_json(files, add_filename=True)
    dataset = orig_dataset

    curation_steps = Sequential(
        [
            clean_and_unify,
            heuristic_filter,
            dedupe,
            redact_pii,
            #quality_filter,
            #decontaminate,
        ]
    )
    print("Executing the pipeline...")

    dataset = curation_steps(dataset)
    dataset = dataset.persist()
    out_path = "/workspace/cc/curated_dumps"
    os.makedirs(out_path, exist_ok=True)
    output_filename = file_name_inp
    output_filepath = os.path.join(out_path, output_filename)
    dataset.to_json(output_filepath, write_to_filename=True)
    print(out_path)
    
def main(file_path,file_name_inp):
    client = get_client(cluster_type = 'gpu')
    client.run(pre_imports)
    process_chunk(file_path,file_name_inp)

    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument('--file', type=str, required=True, help='file to process')
    args = parser.parse_args()

    file_path = args.file 
    file_name_inp = input("Enter the name of the .jsonl file: ")
    start = time.time()
    main(file_path,file_name_inp)
    end = time.time()
    print(f"Time taken: {end - start}s")
