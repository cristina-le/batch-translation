import pandas as pd
import matplotlib.pyplot as plt
from app.temperature_test import run_temperature_test
from app.temperature_test_refine import run_temperature_test_refine

def run_all_tests():
    print("\n" + "="*70)
    print("STARTING TRANSLATION TEMPERATURE TESTS")
    print("="*70)
    
    # Run translation temperature tests
    best_trans_temp, trans_results_df, trans_summary_df = run_temperature_test()
    
    print("\n\n" + "="*70)
    print("STARTING REFINEMENT TEMPERATURE TESTS")
    print("="*70)
    
    # Run refinement temperature tests
    best_refine_temp, refine_results_df, refine_summary_df = run_temperature_test_refine()
    
    # Combine results for visualization
    trans_summary_df['process'] = 'Translation'
    refine_summary_df['process'] = 'Refinement'
    combined_df = pd.concat([trans_summary_df, refine_summary_df])
    
    # Save combined results
    combined_df.to_csv("app/data/combined_temperature_results.csv", index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot translation results
    plt.subplot(2, 1, 1)
    plt.errorbar(
        trans_summary_df['temperature'], 
        trans_summary_df['avg_bleu'],
        yerr=trans_summary_df['std_dev'],
        fmt='o-',
        capsize=5,
        label='Translation'
    )
    plt.axvline(x=best_trans_temp, color='r', linestyle='--', label=f'Best Temp: {best_trans_temp}')
    plt.title('Translation Temperature vs BLEU Score')
    plt.xlabel('Temperature')
    plt.ylabel('Average BLEU Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot refinement results
    plt.subplot(2, 1, 2)
    plt.errorbar(
        refine_summary_df['temperature'], 
        refine_summary_df['avg_bleu'],
        yerr=refine_summary_df['std_dev'],
        fmt='o-',
        capsize=5,
        label='Refinement'
    )
    plt.axvline(x=best_refine_temp, color='r', linestyle='--', label=f'Best Temp: {best_refine_temp}')
    plt.title('Refinement Temperature vs BLEU Score')
    plt.xlabel('Temperature')
    plt.ylabel('Average BLEU Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("app/data/temperature_comparison.png", dpi=300)
    
    print("\n\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Best Translation Temperature: {best_trans_temp}")
    print(f"Best Refinement Temperature: {best_refine_temp}")
    print(f"Results visualization saved to app/data/temperature_comparison.png")
    print(f"Combined results saved to app/data/combined_temperature_results.csv")
    print("="*70)
    
    return best_trans_temp, best_refine_temp, combined_df

if __name__ == "__main__":
    best_trans_temp, best_refine_temp, combined_df = run_all_tests()
