import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from app.utils import preprocess
from app.core.translator import JapaneseToEnglishTranslator
from app.benchmark.calculateBleu import evaluate_translation

load_dotenv()

def run_model_test():
    # File paths
    jp_file = "app/data/batch_jp.txt"
    en_ref_file = "app/data/batch_en.txt"
    results_csv = "app/data/model_test_results.csv"

    # Models to test
    models = ["openai/gpt-4o-mini", "mistralai/mistral-small-3.1-24b-instruct"]
    runs_per_model = 5
    chunk_size = 20
    temperature = 0.3

    # Results storage
    results_data = []

    print(f"Testing {len(models)} models with {runs_per_model} runs each")

    for model in models:
        print(f"\n{'='*50}")
        print(f"Testing model: {model}")
        print(f"{'='*50}")

        model_scores = []

        for run in range(1, runs_per_model + 1):
            print(f"\nRun {run}/{runs_per_model} with model {model}")
            output_file = f"app/data/model_test_output_{model.replace('/', '_')}_{run}.txt"

            # Read Japanese input with fixed chunk size
            japanese_segments = preprocess.reader(jp_file, size=chunk_size)

            # Create translator with current model and fixed temperature
            translator = JapaneseToEnglishTranslator(temperature=temperature, model=model)

            # Translate each segment
            translations = []
            for segment in tqdm(japanese_segments):
                num_lines = len(segment.splitlines())
                translations.append(translator.translate(segment, num_lines))
                time.sleep(0.5)  # Respect API rate limits

            # Write translations to file
            preprocess.writer(output_file, translations)

            # Evaluate BLEU score
            bleu = evaluate_translation(en_ref_file, output_file)
            print(f"Run {run} BLEU score: {bleu:.4f}")

            # Store individual run result
            results_data.append({
                'model': model,
                'run': run,
                'bleu_score': bleu
            })

            model_scores.append(bleu)

    # Create DataFrame from results
    results_df = pd.DataFrame(results_data)

    # Calculate summary statistics
    summary_df = results_df.groupby('model').agg(
        avg_bleu=('bleu_score', 'mean'),
        std_dev=('bleu_score', 'std'),
        min_bleu=('bleu_score', 'min'),
        max_bleu=('bleu_score', 'max')
    ).reset_index()

    # Find best model
    best_model_row = summary_df.loc[summary_df['avg_bleu'].idxmax()]
    best_model = best_model_row['model']

    # Add a column to mark the best model
    summary_df['is_best'] = summary_df['model'] == best_model

    # Save both DataFrames to CSV
    results_df.to_csv(results_csv.replace('.csv', '_detailed.csv'), index=False)
    summary_df.to_csv(results_csv, index=False)

    # Print summary for convenience
    print("\n\n" + "="*60)
    print(f"RESULTS SUMMARY")
    print("="*60)
    print(summary_df.to_string())
    print("\n" + "="*60)
    print(f"Best model: {best_model} with average BLEU score: {best_model_row['avg_bleu']:.4f}")
    print(f"Results saved to {results_csv} and {results_csv.replace('.csv', '_detailed.csv')}")
    print("="*60)

    return best_model, results_df, summary_df

if __name__ == "__main__":
    best_model, results_df, summary_df = run_model_test()
