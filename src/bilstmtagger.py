from collections import Counter
import random,os,codecs,pickle
from optparse import OptionParser
import numpy as np

import util

class Tagger:
    def __init__(self, options, words, tags, bios):
        self.options = options
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify,
                            'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]
        self.vw = util.Vocab.from_corpus([words])
        self.vt = util.Vocab.from_corpus([tags])
        self.vb = util.Vocab.from_corpus([bios])
        self.UNK_W = self.vw.w2i['_UNK_']
        self.UNK_P = self.vt.w2i['_UNK_']

        self.nwords = self.vw.size()
        self.ntags = self.vt.size()
        self.nBios = self.vb.size()

        self.model = Model()
        self.trainer = AdamTrainer(self.model)

        self.WE = self.model.add_lookup_parameters((self.nwords, options.wembedding_dims))
        self.PE = self.model.add_lookup_parameters((self.ntags, options.pembedding_dims))
        self.LE = self.model.add_lookup_parameters((self.nBios+1, options.lembedding_dims)) # label embedding, 1 for start symbol
        self.pH1 = self.model.add_parameters((options.hidden_units, options.his_lstmdims))
        self.pH2 = self.model.add_parameters((options.hidden2_units, options.hidden_units)) if options.hidden2_units>0 else None
        hdim = options.hidden2_units if options.hidden2_units>0 else options.hidden_units
        self.pO = self.model.add_parameters((self.nBios, hdim))

        self.edim = 0
        self.external_embedding = None
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding, 'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                                       external_embedding_fp}
            external_embedding_fp.close()
            self.edim = len(self.external_embedding.values()[0])
            noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.extrn_lookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
            self.extrn_lookup.set_updated(False)
            for word, i in self.extrnd.iteritems():
                self.extrn_lookup.init_row(i, self.external_embedding[word])
                #if self.edim == options.wembedding_dims and self.vw.w2i.has_key(word):
                    #self.WE.init_row(self.vw.w2i.get(word), self.external_embedding[word])
            self.extrnd['_UNK_'] = 1
            self.extrnd['_START_'] = 2
            self.extrn_lookup.init_row(1, noextrn)
            print 'Loaded external embedding. Vector dimensions:', self.edim

        inp_dim = options.wembedding_dims + options.pembedding_dims + self.edim
        history_input_dim = options.lembedding_dims +  2 * options.lstm_dims
        self.builders = [LSTMBuilder(1, inp_dim, options.lstm_dims, self.model),
                         LSTMBuilder(1, inp_dim, options.lstm_dims, self.model)]
        self.history_lstm = LSTMBuilder(1, history_input_dim, options.his_lstmdims, self.model)

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
        hist_init = self.history_lstm.initial_state()
        wembs = [noise(self.WE[w], 0.1) for w in words]
        pembs = [noise(self.PE[t],0.001) for t in tags]
        evec = [self.extrn_lookup[
                    self.extrnd[w]] if self.edim > 0 and w in self.extrnd else self.extrn_lookup[1] if self.edim > 0 else None
                for w in words]
        inputs = [concatenate(filter(None, [wembs[i], pembs[i],evec[i]])) for i in xrange(len(words))]
        fw = [x.output() for x in f_init.add_inputs(inputs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(inputs))]

        H1 = parameter(self.pH1)
        H2 = parameter(self.pH2) if self.pH2!=None else None
        O = parameter(self.pO)
        errs = []

        for f, b, bio, i in zip(fw, reversed(bw), bios, xrange(len(bios))):
            f_b = concatenate([f, b])
            b_i = self.LE[bios[i-1]] if i>0 else self.LE[self.nBios]
            hist_init = hist_init.add_input(concatenate([f_b, b_i]))
            r_bio = O * (self.activation(H2*self.activation(H1 * hist_init.output()))) if H2!=None else O * (self.activation(H1 * hist_init.output()))
            err = pickneglogsoftmax(r_bio, bio)
            errs.append(err)
        return esum(errs)

    def tag_sent(self, sent):
        renew_cg()
        f_init, b_init = [b.initial_state() for b in self.builders]
        hist_init = self.history_lstm.initial_state()
        wembs = [self.WE[self.vw.w2i.get(w, self.UNK_W)] for w, t, bio in sent]
        pembs = [self.PE[self.vt.w2i.get(t, self.UNK_P)] for w, t, bio in sent]
        evec = [self.extrn_lookup[self.extrnd[w]] if self.edim > 0 and  w in self.extrnd else self.extrn_lookup[1] if self.edim>0 else None for w, t, bio in sent]
        inputs = [concatenate(filter(None, [wembs[i], pembs[i], evec[i]])) for i in xrange(len(sent))]
        fw = [x.output() for x in f_init.add_inputs(inputs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(inputs))]

        H1 = parameter(self.pH1)
        H2 = parameter(self.pH2) if self.pH2!=None else None
        O = parameter(self.pO)
        bios = []
        last = None
        for f, b in zip(fw, reversed(bw)):
            f_b = concatenate([f, b])
            b_i = self.LE[last] if last!=None else self.LE[self.nBios]
            hist_init = hist_init.add_input(concatenate([f_b, b_i]))

            r_t = O * (self.activation(H2*self.activation(H1 * hist_init.output()))) if H2!=None else O * (self.activation(H1 * hist_init.output()))
            out = softmax(r_t)
            last = np.argmax(out.npvalue())
            bios.append(self.vb.i2w[last])
        return bios

    def train(self):
        tagged = loss = 0
        best_dev = float('-inf')
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
                        res = good / (good + bad)
                        if res>best_dev:
                            print '\ndev accuracy (saving):', res
                            best_dev = res
                            self.save(os.path.join(options.output, options.model))
                        else:
                            print '\ndev accuracy:', res
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
            dev = list(self.read(options.dev_file))
            good = bad = 0.0
            for sent in dev:
                tags = self.tag_sent(sent)
                golds = [b for w, t, b in sent]
                for go, gu in zip(golds, tags):
                    if go == gu:
                        good += 1
                    else:
                        bad += 1
            res = good / (good + bad)
            if res > best_dev:
                print '\ndev accuracy (saving):', res
                best_dev = res
                self.save(os.path.join(options.output, options.model))
            else:
                print '\ndev accuracy:', res
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
        parser.add_option('--lembedding', type='int', dest='lembedding_dims', default=30)
        parser.add_option('--epochs', type='int', dest='epochs', default=5)
        parser.add_option('--hidden', type='int', dest='hidden_units', default=200)
        parser.add_option('--hidden2', type='int', dest='hidden2_units', default=0)
        parser.add_option('--lstmdims', type='int', dest='lstm_dims', default=200)
        parser.add_option('--his_lstmdims', type='int', dest='his_lstmdims', default=200)
        parser.add_option('--outdir', type='string', dest='output', default='')
        parser.add_option('--outfile', type='string', dest='outfile', default='')
        parser.add_option("--eval", action="store_true", dest="eval_format", default=False)
        parser.add_option("--activation", type="string", dest="activation", default="tanh")
        parser.add_option('--mem', type='int', dest='mem', default=2048)
        return parser.parse_args()

if __name__ == '__main__':
    import _dynet as dy
    (options, args) = Tagger.parse_options()
    dyparams = dy.DynetParams()
    # Fetch the command line arguments (optional)
    dyparams.from_args()
    # Set some parameters manualy (see the command line arguments documentation)
    dyparams.set_mem(options.mem)
    dyparams.init()
    from dynet import *

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

        print 'writing params file'
        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, tags, bios, options), paramsfp)

        Tagger(options, words, tags, bios).train()

        options.model = os.path.join(options.output,options.model+'_'+str(options.epochs-1))
        options.params =  os.path.join(options.output,options.params)

    if options.conll_test != '' and options.params != '' and options.model != '' and options.outfile != '':
        print options.model, options.params, options.eval_format
        print 'reading params'
        with open(options.params, 'r') as paramsfp:
            words, tags, bios, opt = pickle.load(paramsfp)
        tagger = Tagger(opt, words, tags, bios)

        print 'loading model'
        print options.model
        tagger.load(options.model)

        test = list(Tagger.read(options.conll_test))
        print 'loaded',len(test),'sentences!'
        writer = codecs.open(options.outfile, 'w')
        for sent in test:
            output = list()
            tags = tagger.tag_sent(sent)
            if options.eval_format:
                 [output.append(' '.join([sent[i][0], sent[i][1],sent[i][2], tags[i]])) for i in xrange(len(tags))]
            else:
                [output.append(' '.join([sent[i][0],sent[i][1],tags[i]])) for i in xrange(len(tags))]
            writer.write('\n'.join(output))
            writer.write('\n\n')
        print 'done!'