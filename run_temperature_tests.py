import pandas as pd
import matplotlib.pyplot as plt
from app.test.temperature_test import run_model_test

def run_model_analysis():
    print("\n" + "="*70)
    print("STARTING MODEL TESTS")
    print("="*70)

    # Run model tests
    best_model, results_df, summary_df = run_model_test()

    # Save summary results
    summary_df.to_csv("app/data/model_test_results.csv", index=False)

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        summary_df['model'],
        summary_df['avg_bleu'],
        yerr=summary_df['std_dev'],
        fmt='o-',
        capsize=5,
        label='Model'
    )
    plt.axvline(x=best_model, color='r', linestyle='--', label=f'Best Model: {best_model}')
    plt.title('Model vs BLEU Score')
    plt.xlabel('Model')
    plt.ylabel('Average BLEU Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("app/data/model_comparison.png", dpi=300)

    print("\n\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Best Model: {best_model}")
    print(f"Results visualization saved to app/data/model_comparison.png")
    print(f"Summary results saved to app/data/model_test_results.csv")
    print("="*70)

    return best_model, summary_df

if __name__ == "__main__":
    best_model, summary_df = run_model_analysis()
