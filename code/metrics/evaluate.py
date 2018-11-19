import pandas as pda
import bleu
import rouge
import subprocess
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_path")
    parser.add_argument("--config_id")
    args = parser.parse_args()
    return args

def read_results(path,num):
    with open(path+"/labels"+str(num)+".txt","r") as fp:
        l=fp.readlines()
    with open(path+"/predictions"+str(num)+".txt","r") as fp:
        p=fp.readlines()

    return p,l

def exact_match(p,l):
    c=0
    for i1,i in enumerate(l):
        if p[i1]==l[i1]:
            c+=1
    print("Exact Match: ",c/len(l))


def moses_bl_rouge(p,l):
    bl = bleu.moses_multi_bleu(p,l)
    x = rouge.rouge(p,l)
    print('Moses BLEU: %f\nROUGE1-F: %f\nROUGE1-P: %f\nROUGE1-R: %f\nROUGE2-F: %f\nROUGE2-P: %f\nROUGE2-R: %f\nROUGEL-F: %f\nROUGEL-P: %f\nROUGEL-R: %f'%(bl,x['rouge_1/f_score'],x['rouge_1/p_score'],x['rouge_1/r_score'],x['rouge_2/f_score'],
                                                    x['rouge_2/p_score'],x['rouge_2/r_score'],x['rouge_l/f_score'],x['rouge_l/p_score'],x['rouge_l/r_score']))

if __name__=='__main__':
    args = get_args()
    result_path = args.preds_path
    config_id = args.config_id
    preds,labels = read_results(result_path,config_id)
    exact_match(preds,labels)
    moses_bl_rouge(preds,labels)

