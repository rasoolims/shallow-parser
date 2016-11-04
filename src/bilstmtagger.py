from dynet import *
from collections import Counter
import random,sys,os,codecs,pickle
from optparse import OptionParser
import numpy as np

import util

class Tagger:
    def __init__(self, options, words, tags, bios):
        self.options = options
        self.MLP = True
        self.vw = util.Vocab.from_corpus([words])
        self.vt = util.Vocab.from_corpus([tags])
        self.vb = util.Vocab.from_corpus([bios])
        self.UNK_W = self.vw.w2i['_UNK_']
        self.UNK_P = self.vt.w2i['_UNK_']

        self.nwords = self.vw.size()
        self.ntags = self.vt.size()
        self.nBios = self.vb.size()

        print 'writing params file'
        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, tags, bios, options), paramsfp)

        self.model = Model()
        self.trainer = AdamTrainer(self.model)

        self.WE = self.model.add_lookup_parameters((self.nwords, options.wembedding_dims))
        self.PE = self.model.add_lookup_parameters((self.ntags, options.pembedding_dims))
        if self.MLP:
            self.pH = self.model.add_parameters((options.hidden_units, options.lstm_dims * 2))
            self.pO = self.model.add_parameters((self.ntags, options.hidden_units))
        else:
            self.pO = self.model.add_parameters((self.ntags, options.lstm_dims * 2))

        inp_dim = options.wembedding_dims + options.pembedding_dims
        self.builders = [LSTMBuilder(1, inp_dim, options.lstm_dims, self.model),
                         LSTMBuilder(1, inp_dim, options.lstm_dims, self.model)]

    @staticmethod
    def write(sentences,fname):
        writer = codecs.open(fname,'w')
        for sent in sentences:
            for s in sent:
                writer.write('\t'.join(s))
                writer.write('\n')
            writer.write('\n')
        writer.close()

    @staticmethod
    def read(fname):
        sent = []
        for line in file(fname):
            line = line.strip().split()
            if not line:
                if sent: yield sent
                sent = []
            else:
                w, p, bio = line
                sent.append((w, p, bio))
        if sent: yield  sent

    def build_tagging_graph(self, words, tags, bios):
        renew_cg()
        f_init, b_init = [b.initial_state() for b in self.builders]
        wembs = [noise(self.WE[w], 0.1) for w in words]
        pembs = [noise(self.PE[t],0.001) for t in tags]
        inputs = [concatenate([wembs[i], pembs[i]]) for i in xrange(len(words))]
        fw = [x.output() for x in f_init.add_inputs(inputs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(inputs))]

        if self.MLP:
            H = parameter(self.pH)
            O = parameter(self.pO)
        else:
            O = parameter(self.pO)
        errs = []
        for f, b, bio in zip(fw, reversed(bw), bios):
            f_b = concatenate([f, b])
            if self.MLP:
                r_bio = O * (tanh(H * f_b))
            else:
                r_bio = O * f_b
            err = pickneglogsoftmax(r_bio, bio)
            errs.append(err)
        return esum(errs)

    def tag_sent(self, sent):
        renew_cg()
        f_init, b_init = [b.initial_state() for b in self.builders]
        wembs = [self.WE[self.vw.w2i.get(w, self.UNK_W)] for w, t, bio in sent]
        pembs = [self.PE[self.vt.w2i.get(t, self.UNK_P)] for w, t, bio in sent]
        inputs = [concatenate([wembs[i], pembs[i]]) for i in xrange(len(sent))]
        fw = [x.output() for x in f_init.add_inputs(inputs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(inputs))]

        if self.MLP:
            H = parameter(self.pH)
            O = parameter(self.pO)
        else:
            O = parameter(self.pO)
        bios = []
        for f, b in zip(fw, reversed(bw)):
            if self.MLP:
                r_t = O * (tanh(H * concatenate([f, b])))
            else:
                r_t = O * concatenate([f, b])
            out = softmax(r_t)
            chosen = np.argmax(out.npvalue())
            bios.append(self.vb.i2w[chosen])
        return bios

    def train(self):
        tagged = loss = 0
        for ITER in xrange(self.options.epochs):
            print 'ITER', ITER
            random.shuffle(train)
            for i, s in enumerate(train, 1):
                if i % 1000 == 0:
                    self.trainer.status()
                    print loss / tagged
                    loss = 0
                    tagged = 0

                    good = bad = 0.0
                    if options.dev_file:
                        dev = list(self.read(options.dev_file))
                        for sent in dev:
                            tags = self.tag_sent(sent)
                            golds = [b for w, t, b in sent]
                            for go, gu in zip(golds, tags):
                                if go == gu:
                                    good += 1
                                else:
                                    bad += 1
                        print '\ndev accuracy:', good / (good + bad)
                ws = [self.vw.w2i.get(w, self.UNK_W) for w, p, bio in s]
                ps = [self.vt.w2i[p] for w, p, bio in s]
                bs = [self.vb.w2i[bio] for w, p, bio in s]
                sum_errs = self.build_tagging_graph(ws, ps, bs)
                squared = -sum_errs  # * sum_errs
                loss += sum_errs.scalar_value()
                tagged += len(ps)
                sum_errs.backward()
                self.trainer.update()
            print 'saving current iteration'
            self.save(os.path.join(options.output, options.model + '_' + str(ITER)))

    def load(self, f):
        self.model.load(f)

    def save(self, f):
        self.model.save(f)

    @staticmethod
    def parse_options():
        parser = OptionParser()
        parser.add_option('--train', dest='conll_train', help='Annotated CONLL train file', metavar='FILE', default='')
        parser.add_option('--dev', dest='dev_file', help='Annotated CONLL development file', metavar='FILE', default='')
        parser.add_option('--test', dest='conll_test', help='Annotated CONLL test file', metavar='FILE', default='')
        parser.add_option('--params', dest='params', help='Parameters file', metavar='FILE', default='params.pickle')
        parser.add_option('--extrn', dest='external_embedding', help='External embeddings', metavar='FILE')
        parser.add_option('--model', dest='model', help='Load/Save model file', metavar='FILE', default='model.model')
        parser.add_option('--wembedding', type='int', dest='wembedding_dims', default=128)
        parser.add_option('--pembedding', type='int', dest='pembedding_dims', default=30)
        parser.add_option('--epochs', type='int', dest='epochs', default=5)
        parser.add_option('--hidden', type='int', dest='hidden_units', default=200)
        parser.add_option('--hidden2', type='int', dest='hidden2_units', default=0)
        parser.add_option('--lstmdims', type='int', dest='lstm_dims', default=200)
        parser.add_option('--outdir', type='string', dest='output', default='')
        parser.add_option('--outfile', type='string', dest='outfile', default='')
        return parser.parse_args()

#todo add external embeddings
if __name__ == '__main__':
    (options, args) = Tagger.parse_options()
    if options.conll_train != '' and options.output != '':
        train = list(Tagger.read(options.conll_train))
        print 'load #sent:',len(train)
        words = []
        tags = []
        bios = []
        wc = Counter()
        for s in train:
            for w, p, bio in s:
                words.append(w)
                tags.append(p)
                bios.append(bio)
                wc[w] += 1
        words.append('_UNK_')
        tags.append('_UNK_')
        tags.append('_START_')
        bios.append('_START_')

        Tagger(options, words, tags, bios).train()

        options.model = os.path.join(options.output,options.model+'_'+str(options.epochs-1))
        options.params =  os.path.join(options.output,options.params)

    if options.conll_test != '' and options.params != '' and options.model != '' and options.outfile != '':
        print options.model, options.params
        print 'reading params'
        with open(options.params, 'r') as paramsfp:
            words, tags, bios, opt = pickle.load(paramsfp)
        tagger = Tagger(opt, words, tags, bios)

        print 'loading model'
        print options.model
        tagger.load(options.model)

        test = list(Tagger.read(options.conll_test))
        print 'loaded',len(test),'sentences!'

        for sent in test:
            tags = tagger.tag_sent(sent)
            sent = [(sent[i][0],sent[i][1],tags[i]) for i in xrange(len(tags))]
        print 'writing predictions'
        Tagger.write(test, options.outfile)
        print 'done!'