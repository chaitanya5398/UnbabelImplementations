#gives the pearson rho of the predictions.
#Running instruction: python hterevaluate.py  <name_of_prediction_file>

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import sys

ref_hter='/home/krishna/Summarizartion/TQE/data/t2/train/dev.hter'
pred_hter=sys.argv[1]

def load_scores(fname):
    ret=[]
    with open(fname,'r') as fp:
        for j in fp:
            j = j.strip()
            j = float(j)
            ret.append(j)
    return ret

if __name__=='__main__':
    true = load_scores(ref_hter)
    pred = load_scores(pred_hter)
    pr = pearsonr(true,pred)
    mer = mae(true,pred)
    mser = mse(true,pred)
    print "Pearson Rho => ",pr
    print "Mean Absolute Error => ",mer
    print "Mean Squared Error => ",mser
    
