import sacrebleu
from sacremoses import MosesDetokenizer

def evaluate_translation(target_test, target_pred):
    """
    Evaluate BLEU score between reference and predicted translation files.
    """
    md = MosesDetokenizer(lang='en')
    refs = []

    with open(target_test, encoding="utf8") as test:
        for line in test: 
            line = line.strip().split() 
            line = md.detokenize(line) 
            refs.append(line)
        
    refs = [refs]

    preds = []

    with open(target_pred, encoding="utf8") as pred:  
        for line in pred: 
            line = line.strip().split() 
            line = md.detokenize(line) 
            preds.append(line)

    bleu = sacrebleu.corpus_bleu(preds, refs)
    return bleu.score