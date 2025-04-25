import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from app.utils import preprocess
from app.core.refiner import TranslationRefiner
from app.benchmark.calculateBleu import evaluate_translation

load_dotenv()

def run_temperature_test_refine():
    """
    Run temperature sweep for translation refinement and analyze BLEU scores.

    Returns:
        tuple: (best_temp, results_df, summary_df)
    """
    jp_file = "app/data/batch_jp.txt"
    en_ref_file = "app/data/batch_en.txt"
    initial_pred_file = "app/data/batch_output.txt"
    results_csv = "app/data/temperature_test_refine_results.csv"
    
    temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    runs_per_temp = 5
    model = "google/gemini-2.5-pro-preview-03-25"
    chunk_size = 50
    
    results_data = []
    
    japanese_segments = preprocess.reader(jp_file, size=chunk_size)
    current_translations = preprocess.reader(initial_pred_file, size=chunk_size)
    
    print(f"Testing {len(temperatures)} temperature values with {runs_per_temp} runs each")
    
    for temp in temperatures:
        print(f"\n{'='*50}")
        print(f"Testing temperature: {temp}")
        print(f"{'='*50}")
        
        temp_scores = []
        
        for run in range(1, runs_per_temp + 1):
            print(f"\nRun {run}/{runs_per_temp} with temperature {temp}")
            output_file = f"app/data/temp_test_refine_output_{temp}_{run}.txt"
            
            refiner = TranslationRefiner(temperature=temp, model=model)
            
            refined = []
            for (jp, cur) in tqdm(zip(japanese_segments, current_translations)):
                num_lines = len(cur.splitlines())
                refined.append(refiner.refine(jp, cur, num_lines))
                time.sleep(0.5)
            
            preprocess.writer(output_file, refined)
            
            bleu = evaluate_translation(en_ref_file, output_file)
            print(f"Run {run} BLEU score: {bleu:.4f}")
            
            results_data.append({
                'temperature': temp,
                'run': run,
                'bleu_score': bleu
            })
            
            temp_scores.append(bleu)
    
    results_df = pd.DataFrame(results_data)
    
    summary_df = results_df.groupby('temperature').agg(
        avg_bleu=('bleu_score', 'mean'),
        std_dev=('bleu_score', 'std'),
        min_bleu=('bleu_score', 'min'),
        max_bleu=('bleu_score', 'max')
    ).reset_index()
    
    best_temp_row = summary_df.loc[summary_df['avg_bleu'].idxmax()]
    best_temp = best_temp_row['temperature']
    
    summary_df['is_best'] = summary_df['temperature'] == best_temp
    
    results_df.to_csv(results_csv.replace('.csv', '_detailed.csv'), index=False)
    summary_df.to_csv(results_csv, index=False)
    
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
    best_temp, results_df, summary_df = run_temperature_test_refine()
