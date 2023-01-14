#coding=utf8

import sys, os, time, gc, json
from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)


model = SLUTagging(args).to(device)
Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)

def predict():
    model.eval()
    test_path = os.path.join(args.dataroot, 'test.json')
    test_dataset = Example.load_dataset(test_path)
    predictions = []
    did = []
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=False)
            pred = model.decode(Example.label_vocab, current_batch)
            predictions.extend(pred)
            did.extend(current_batch.did)
    test_json = json.load(open(test_path, 'r', encoding="utf8"))
    ptr = 0
    for example in test_json[:10]:
        for utt in example:
            utt['pred'] = [pred.split('-') for pred in predictions[did.index(ptr)]]
            ptr += 1
    json.dump(test_json, open(os.path.join(args.dataroot, 'test.json'), 'w', encoding='utf8'), indent=4, ensure_ascii=False)

check_point = torch.load(open('model.bin', 'rb'), map_location=device)
model.load_state_dict(check_point['model'])
print("Load saved model from root path")
predict()
print("Prediction accomplished. Output saved in file 'test.json'")