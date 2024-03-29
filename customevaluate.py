#gives the f1 score of the predictions.
#Running instruction: python custimevaluate.py <name_of_prediction_file>
import sys
from sklearn.metrics import f1_score

ref_tags='/home/krishna/Summarizartion/TQE/data/t2/train/dev.tags'
pred_tags=sys.argv[1]

def generate_labels(fname):
    fl=[]
    with open(fname,'r') as fp:
        for j in fp:
            j = j.split()
            for k in j:
                if k=='OK':
                    fl.append(1)
                elif k=='BAD':
                    fl.append(0)
                else:
                    print "Wrong man"
    print len(fl)
    return fl

if __name__=='__main__':
    flat_true = generate_labels(ref_tags)
    flat_pred = generate_labels(pred_tags)
    f1_bad, f1_good = f1_score(flat_true, flat_pred, average=None, pos_label=None)
    print("F1-BAD: ", f1_bad, "F1-OK: ", f1_good)
    print("F1-score multiplied: ", f1_bad * f1_good)
