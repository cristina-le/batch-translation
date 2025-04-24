import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from app.utils import preprocess
from app.core.translator import JapaneseToEnglishTranslator
from app.benchmark.calculateBleu import evaluate_translation

load_dotenv()

def run_temperature_test():
    # File paths
    jp_file = "app/data/japaneseOriginal.txt"
    en_ref_file = "app/data/humanTranslation.txt"
    results_csv = "app/data/temperature_test_results.csv"
    
    # Temperature values to test
    temperatures = [ 0.1, 0.2, 0.3]
    runs_per_temp = 3
    model = "google/gemini-2.5-flash-preview"
    chunk_size = 50
    
    # Results storage
    results_data = []
    
    # Read Japanese input
    japanese_segments = preprocess.reader(jp_file, size=chunk_size)
    
    print(f"Testing {len(temperatures)} temperature values with {runs_per_temp} runs each")
    
    for temp in temperatures:
        print(f"\n{'='*50}")
        print(f"Testing temperature: {temp}")
        print(f"{'='*50}")
        
        temp_scores = []
        
        for run in range(1, runs_per_temp + 1):
            print(f"\nRun {run}/{runs_per_temp} with temperature {temp}")
            output_file = f"app/data/temp_test_output_{temp}_{run}.txt"
            
            # Create translator with current temperature
            translator = JapaneseToEnglishTranslator(temperature=temp, model=model)
            
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
                'temperature': temp,
                'run': run,
                'bleu_score': bleu
            })
            
            temp_scores.append(bleu)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results_data)
    
    # Calculate summary statistics
    summary_df = results_df.groupby('temperature').agg(
        avg_bleu=('bleu_score', 'mean'),
        std_dev=('bleu_score', 'std'),
        min_bleu=('bleu_score', 'min'),
        max_bleu=('bleu_score', 'max')
    ).reset_index()
    
    # Find best temperature
    best_temp_row = summary_df.loc[summary_df['avg_bleu'].idxmax()]
    best_temp = best_temp_row['temperature']
    
    # Add a column to mark the best temperature
    summary_df['is_best'] = summary_df['temperature'] == best_temp
    
    # Save both DataFrames to CSV
    results_df.to_csv(results_csv.replace('.csv', '_detailed.csv'), index=False)
    summary_df.to_csv(results_csv, index=False)
    
    # Print summary for convenience
    print("\n\n" + "="*60)
    print(f"RESULTS SUMMARY")
    print("="*60)
    print(summary_df.to_string())
    print("\n" + "="*60)
    print(f"Best temperature: {best_temp} with average BLEU score: {best_temp_row['avg_bleu']:.4f}")
    print(f"Results saved to {results_csv} and {results_csv.replace('.csv', '_detailed.csv')}")
    print("="*60)
    
    return best_temp, results_df, summary_df

if __name__ == "__main__":
    best_temp, results_df, summary_df = run_temperature_test()
