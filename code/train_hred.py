import tensorflow as tf
import numpy as np
import json
import os
import pickle

from tqdm import tqdm
from hred.model import HREDModel
from hred.data_utils import pad, replace_token_no, get_len

flags = tf.app.flags
flags.DEFINE_string("config_id",'60',"Hyperparam config id")
flags.DEFINE_string("data_dir", "../data/english", "Data directory ")
flags.DEFINE_string("infer_data", "test", "[train, dev or test]")
flags.DEFINE_string("logs_dir", "logs", "Data directory ")
flags.DEFINE_string("checkpoint_dir", 'checkpoints', "checkpoint directory")
flags.DEFINE_string("rnn_unit", 'gru', "GRU or LSTM")
flags.DEFINE_float("learning_rate", 0.0004, "Learning rate for Adam Optimizer")
flags.DEFINE_integer("batch_size", 32, "Batch size for training")
flags.DEFINE_integer("epochs",50, "Number of epochs to train for")
flags.DEFINE_integer("max_gradient_norm", 10, "Max grad norm. 0 for no clipping")
flags.DEFINE_float("dropout", 0.75, "keep probability for keeping unit")
flags.DEFINE_integer("num_layers", 1, "No of layers of stacking in RNN")
flags.DEFINE_integer("word_emb_dim",300, "hidden dimensions of the word embeddings.")
flags.DEFINE_integer("hidden_units",350, "hidden dimensions of the decoder units.")
flags.DEFINE_integer("eval_interval", 1, "After how many epochs do you want to eval test set")
flags.DEFINE_integer("patience", 5, "Patience parameter")
flags.DEFINE_boolean("train",True,"Train or Infer")
flags.DEFINE_boolean("debug",True,"Debug mode with small dataset")
FLAGS = flags.FLAGS

def arrayfy(data, stats, vocab):
    """
    Create data-arrays from the nested-list form of data
        data: The data in nested list form
        stats: The stats file dumped from preprocessing
        vocab: The vocab file dumped from preprocessing
    """
    context_len,dec_ip_len,dec_op_len,sent_len=get_len(data)
    pad(data,stats)
    replace_token_no(data,vocab)

    context=np.asarray(data[0])
    dec_ip_arr=np.asarray(data[1])
    dec_op_arr=np.asarray(data[2])
    context_len_arr=np.asarray(context_len)
    dec_ip_len_arr=np.asarray(dec_ip_len)
    target_l_arr=[]
    for i in range(len(data[2])):
        fill=list(np.zeros(stats[2],dtype=int))
        for j in range(dec_op_len[i]):
            fill[j]=1
        target_l_arr.append(fill)

    target_len_arr=np.asarray(target_l_arr)

    for i in sent_len:
        for j in range(stats[0]-len(i)):
            i.append(0)

    sent_len_arr=np.asarray(sent_len)

    return [context,dec_ip_arr,dec_op_arr,context_len_arr,dec_ip_len_arr,target_len_arr,sent_len_arr]

def read_data(directory):
    """
    Read the data and associated files from the
    data directory and return it in the form of arrays

    args:
        directory: The data directory specified by FLAGS.data_dir
    """
    if not os.path.exists(FLAGS.logs_dir+FLAGS.config_id+'/'):
        os.mkdir(FLAGS.logs_dir+FLAGS.config_id+'/')

    # with open(directory+'/phred-dialog-dstc2-train.json','r') as fp:
    #     train_data=json.load(fp)
    #     print(len(train_data))
    # with open(directory+'/phred-dialog-dstc2-test.json','r') as fp:
    #     test_data=json.load(fp)
    # with open(directory+'/phred-dialog-dstc2-dev.json','r') as fp:
    #     dev_data=json.load(fp)
    # with open(directory+'/phred-dialog-dstc2-stats.json','r') as fp:
    #     stats=json.load(fp)
    # with open(directory+'/phred-dialog-dstc2-vocab.json','r') as fp:
    #     vocab=json.load(fp)

    with open(directory+'/train.json','r') as fp:
        train_data=json.load(fp)
        print(len(train_data))
    with open(directory+'/test.json','r') as fp:
        test_data=json.load(fp)
    with open(directory+'/dev.json','r') as fp:
        dev_data=json.load(fp)
    # with open(directory+'/stats.json','r') as fp:
    #     stats=json.load(fp)
    with open(directory+'/words.pkl','rb') as fp:
        words = pickle.load(fp)
        vocab = {}
        for i, key in enumerate(words.keys()):
            vocab[key] = i

    def get_dec_outputs(data):

        dec_op=[]
        for i in data[1]:
            temp=i+['<EOS>']
            temp=temp[1:]
            dec_op.append(temp)
        data.append(dec_op)

    get_dec_outputs(train_data)
    get_dec_outputs(dev_data)
    get_dec_outputs(test_data)


    def data_stats(data):

        for ind,d in enumerate(data):
            if ind==0: #pre
                c_len=[]
                for context in d:
                    c_len.append(len(context))
            if ind==1: #KB
                utt_len=[]
                for context in d:
                    utt_len.append(len(context))
            if ind==2: #post
                resp_len=[]
                for context in d:
                    resp_len.append(len(context))

        utterances_len=utt_len+resp_len

        return [max(c_len),max(utterances_len),max(utterances_len)]

    train_stats=data_stats(train_data)
    test_stats=data_stats(test_data)
    dev_stats=data_stats(dev_data)

    stats=[max(test_stats[0],max(train_stats[0],dev_stats[0])),
                  max(test_stats[1],max(train_stats[1],dev_stats[1])),
                 max(test_stats[2],max(train_stats[2],dev_stats[2]))]

    params_dict=FLAGS.__flags
    params_dict['max_enc_size']=stats[0]
    params_dict['max_sent_size']=stats[1]
    params_dict['vocab_size']=len(vocab)

    train=arrayfy(train_data,stats,vocab)
    test=arrayfy(test_data,stats,vocab)
    dev=arrayfy(dev_data,stats,vocab)

    return train,test,dev

def create_model(sess,FLAGS):
    """
    Create a new model if there are no checkpoints
    otherwise restore the model from the existing
    checkpoint

    args:
        sess: The active Session
        FLAGS: The configuration FLAGS
    """
    print("Creating/Restoring HRED Model.....")
    model = HREDModel(sess,FLAGS)

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir+FLAGS.config_id)
    if ckpt:
        print("Restoring model parameters from %s" %
              ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created new model!")
        sess.run(tf.global_variables_initializer())

    return model


def save_model(sess,model):
    """
    Save the model in the checkpoint directory

    args:
        sess: The active Session
        model: The model object which is created or restored
    """
    if not os.path.exists(FLAGS.checkpoint_dir+FLAGS.config_id):
        os.makedirs(FLAGS.checkpoint_dir+FLAGS.config_id)
    save_path = model.saver.save(sess, os.path.join(FLAGS.checkpoint_dir+FLAGS.config_id, "model.ckpt"))
    print("Model saved in file: %s" % save_path)


def get_words_from_ids(ids):
    """
    Convert token ids to the corresponding words by looking up
    from the vocabulary file. It breaks the generated sentence
    at the first '<EOS>' token.

    arg:
        ids: The predicted ids obtained from argmax over the Vocab softmax values
    """
    ids_list=ids.tolist()
    with open(FLAGS.data_dir+'/phred-dialog-dstc2-vocab.json','r') as fp:
         vocab=json.load(fp)

    invert_vocab= dict([v,k] for k,v in vocab.items())

    r=[]
    for i in ids_list:
        c=''
        for j in i:
            c=c+' '+invert_vocab[j]
            if invert_vocab[j]=='<EOS>':
                break
        r.append(c.strip())

    return r

def eval_(data,sess,model,FLAGS,epoch):
    """
    Run one pass over the validation set
    to get the average validation loss

    args:
        data: The whole data in the array format
        sess: The active Session
        model: The model object which has been created or restored
        FLAGS: The configuration FLAGS
        epoch: The current epoch number
    """
    #setup data batch indices
    batch_size = FLAGS.batch_size
    num_ex = data[0].shape[0]
    batches = zip(range(0, num_ex, batch_size), range(batch_size, num_ex+batch_size, batch_size))
    batches = [(start, end) for start, end in batches]

    #Start forward pass on the dev batches
    losses=[]
    preds_all=np.zeros(FLAGS.max_sent_size)
    for i,j in batches:
        batch_data=[data[k][i:j] for k in range(len(data))]
        pred,loss,_ =model.step(sess,FLAGS,batch_data,True,1.0)
        preds_all = np.row_stack((preds_all,pred))
        losses.append(loss)

    avg_loss=np.mean(losses)
    return avg_loss


def train():
    """
    Set up batches of the data and run training on them.
    Also collects the validation losses after
    FLAGS.eval_interval number of epochs. Logs them in the FLAGS.logs_dir
    """
    print("Reading Dataset....")
    dir_=FLAGS.data_dir
    train_examples, test_examples, dev_examples = read_data(dir_)

    # If in debugging mode then run the training of 2 epochs with a smaller data of 67 examples only
    if FLAGS.debug==True:
        train_examples=[train_examples[k][0:67] for k in range(len(train_examples))]
        dev_examples=[dev_examples[k][0:67] for k in range(len(dev_examples))]
        FLAGS.epochs=2

    print("Finished Reading Dataset!")

    #setup data batch indices
    batch_size = FLAGS.batch_size
    num_train = train_examples[0].shape[0]
    batches = zip(range(0, num_train, batch_size), range(batch_size,num_train+batch_size, batch_size))
    batches = [(start, end) for start, end in batches]
    fp=open(FLAGS.logs_dir+FLAGS.config_id+'/logs'+FLAGS.config_id+'.log','w+')


    with tf.Session() as sess:
            #Create or Restore Model
            model=create_model(sess,FLAGS)
            try:
                #Run Training
                for epoch in range(1,FLAGS.epochs+1):
                    train_loss=[]

                    for i,j in tqdm(batches):
                        batch_train =[train_examples[k][i:j] for k in range(len(train_examples))]
                        ypred, loss,_ =model.step(sess,FLAGS,batch_train,False,FLAGS.dropout)
                        train_loss.append(loss)
                        fp.write("Epoch:"+str(epoch)+" batch train loss: "+str(loss)+'\n')

                    print("Epoch: ",epoch," Train loss: ",np.mean(train_loss))
                    if epoch>0 and epoch % FLAGS.eval_interval==0:
                        val_loss=eval_(dev_examples,sess,model,FLAGS,epoch)
                        print("Val Loss: "+str(val_loss)+" Train loss: "+str(np.mean(train_loss)))
                        fp.write("Val Loss: "+str(val_loss)+"Train loss: "+str(np.mean(train_loss))+'\n\n\n\n')

                        print('Saving Model...')
                        fp.write("Saving Model\n")
                        save_model(sess,model)

            except KeyboardInterrupt:
                print("Keyboard Interrupt")
            finally:
                fp.close()



def infer(data_infer):
    """
    Run inference on the dataset specified.
    It dumps the generated sentences and the ground truth sentences.

    args:
        data_infer: The dataset on which inference is going to be performed.
    """

    dir_=FLAGS.data_dir
    train_examples, test_examples, dev_examples = read_data(dir_)

    if data_infer=='test':
        data=test_examples
    elif data_infer=='dev':
        data=dev_examples
    elif data_infer=='train':
        data=train_examples
    else:
        print("Invalid Choice!!")
        return

    # If debugging mode is on then run inference only on a smaller dataset of 67 examples
    if FLAGS.debug:
        data = [data[k][0:67] for k in range(len(data))]

    #set up batch indices
    batch_size = FLAGS.batch_size
    num_ex = data[0].shape[0]
    batches = zip(range(0, num_ex, batch_size), range(batch_size, num_ex+batch_size, batch_size))
    batches = [(start, end) for start, end in batches]

    with tf.Session(graph=tf.Graph()) as sess:
            model=create_model(sess,FLAGS)
            all_wts=[]
            preds_all=np.zeros(FLAGS.max_sent_size)
            for i,j in tqdm(batches):
                batch_data=[data[k][i:j] for k in range(len(data))]
                pred,loss,wts =model.step(sess,FLAGS,batch_data,True,1.0)
                all_wts.append(wts)
                preds_all = np.row_stack((preds_all,pred))


    preds_ids=np.delete(preds_all,0,0)
    preds_test=get_words_from_ids(preds_ids)
    labels_test=get_words_from_ids(data[2])

    os.makedirs("Results")
    fp1 =open('Results/predictions'+str(FLAGS.config_id)+'.txt','w+')
    for item in preds_test:
        fp1.write("%s\n"%item)
    fp1.close()

    fp2 =open('Results/labels'+str(FLAGS.config_id)+'.txt','w+')
    for item in labels_test:
        fp2.write("%s\n"%item)
    fp2.close()

def main():
    if FLAGS.train:
        train()
        FLAGS.train=False
        infer(FLAGS.infer_data)
    else:
        infer(FLAGS.infer_data)


if __name__=='__main__':
    main()
