import sacrebleu
from sacremoses import MosesDetokenizer

def evaluate_translation(target_test, target_pred):
    md = MosesDetokenizer(lang='en')
    # Open the test dataset human translation file and detokenize the references
    refs = []

    with open(target_test, encoding="utf8") as test:
        for line in test: 
            line = line.strip().split() 
            line = md.detokenize(line) 
            refs.append(line)
        
    refs = [refs]  # Yes, it is a list of list(s) as required by sacreBLEU

    # Open the translation file by the NMT model and detokenize the predictions
    preds = []

    with open(target_pred, encoding="utf8") as pred:  
        for line in pred: 
            line = line.strip().split() 
            line = md.detokenize(line) 
            preds.append(line)

    # Calculate and print the BLEU score
    bleu = sacrebleu.corpus_bleu(preds, refs)
    return bleu.score