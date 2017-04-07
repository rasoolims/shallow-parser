from collections import Counter
import random,os,codecs,pickle,time
from optparse import OptionParser
import util
from postagger import Tagger

class Chunker(Tagger):
    def __init__(self, options, vw, vt, vc, vb):
        Tagger.__init__(self,options,vw,vt,vc)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify}
        self.activation = self.activations[options.activation]
        self.vb = vb
        self.UNK_W = self.vw.w2i['_UNK_']
        self.nBios = self.vb.size()
        print 'num of bio tags',self.nBios
        self.PE = self.model.add_lookup_parameters((self.ntags, options.pembedding_dims))
        self.H1 = self.model.add_parameters((options.hidden_units, options.lstm_dims)) if options.hidden_units > 0 else None
        self.H2 = self.model.add_parameters((options.hidden2_units, options.hidden_units)) if options.hidden2_units > 0 else None
        hdim = options.hidden2_units if options.hidden2_units>0 else options.hidden_units if options.hidden_units>0 else options.lstm_dims
        self.O = self.model.add_parameters((self.nBios, hdim))
        self.transitions = self.model.add_lookup_parameters((self.nBios, self.nBios))
        inp_dim = options.wembedding_dims + self.edim + options.clstm_dims + self.ntags + options.pembedding_dims
        self.chunk_lstms = BiRNNBuilder(self.k, inp_dim, options.lstm_dims, self.model, LSTMBuilder if not options.gru else GRUBuilder)

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

    @staticmethod
    def read_raw_file(fname):
        for line in file(fname):
            sent = line.strip().split()
            yield sent

    def build_graph(self, sent_words, words, is_train):
        input_lstm = self.get_lstm_features(is_train, sent_words, words, False)[0]

        O = parameter(self.tagO)
        probs = []
        for f in input_lstm:
            score_t = O*f
            probs.append(score_t)
        return probs

    def get_lstm_features(self, is_train, sent_words, words, is_chunking, predicated_pos = None):
        if is_chunking:
            tag_lstm, char_lstms, wembs, evec = self.get_lstm_features(is_train, sent_words, words, False)
            pembs = [noise(self.PE[p], 0.001) if is_train else self.PE[p] for p in predicated_pos]
            O = parameter(self.tagO)
            tag_scores = []
            for f in tag_lstm:
                score_t = O * f
                tag_scores.append(score_t)
            inputs = [concatenate(filter(None, [wembs[i], evec[i], char_lstms[i][-1], softmax(tag_scores[i]),pembs[i]])) for i in xrange(len(words))]
            input_lstm = self.chunk_lstms.transduce(inputs)
            return input_lstm,tag_scores
        else:
            char_lstms = []
            for w in sent_words:
                char_lstms.append(self.char_lstms.transduce([self.CE[self.vc.w2i[c]] if (c in self.vc.w2i and not is_train) or (is_train and random.random() >= 0.001) else self.CE[self.vc.w2i[' ']] for c in ['<s>'] + list(w) + ['</s>']]))
            wembs = [noise(self.WE[w], 0.1) if is_train else self.WE[w] for w in words]
            evec = [self.extrn_lookup[self.extrnd[w]] if self.edim > 0 and w in self.extrnd else self.extrn_lookup[1] if self.edim > 0 else None for w in words]
            inputs = [concatenate(filter(None, [wembs[i], evec[i], char_lstms[i][-1]])) for i in xrange(len(words))]
            if self.drop:
                [dropout(inputs[i], self.dropout) for i in xrange(len(inputs))]
            input_lstm = self.tag_lstms.transduce(inputs)
            return input_lstm, char_lstms, wembs, evec

    def build_graph(self, sent_words, words, is_train, predicated_pos):
        input_lstm,pos_probs = self.get_lstm_features(is_train, sent_words, words, True, predicated_pos)
        H1 = parameter(self.H1) if self.H1 != None else None
        H2 = parameter(self.H2) if self.H2 != None else None
        O = parameter(self.O)
        scores = []

        for f in input_lstm:
            score_t = O*(self.activation(H2*self.activation(H1 * f))) if H2!=None else O * (self.activation(H1 * f)) if self.H1 != None  else O * f
            scores.append(score_t)
        return scores,pos_probs

    def neg_log_loss(self, sent_words, words, labels, predicated_pos):
        observations = self.build_graph(sent_words, words, True, predicated_pos)[0]
        gold_score = self.score_sentence(observations, labels, self.transitions, self.vb.w2i)
        forward_score = self.forward(observations, self.nBios, self.transitions, self.vb.w2i)
        return forward_score - gold_score

    def tag_sent(self, sent, pos_tagger):
        renew_cg()
        words = [w for w, p, bio in sent]
        ws = [self.vw.w2i.get(w, self.UNK_W) for w, p, bio in sent]
        pos_tags = None
        if pos_tagger:
            tag_scores = pos_tagger.get_tag_scores(False,words, ws)
            pos_tags, _ = pos_tagger.viterbi_decoding(tag_scores, pos_tagger.tag_transitions, pos_tagger.vt.w2i, pos_tagger.ntags)
        observations,tag_scores = self.build_graph(words, ws, False, pos_tags)
        bios, score = self.viterbi_decoding(observations,self.transitions,self.vb.w2i, self.nBios)
        if not pos_tagger:
            pos_tags, _ = self.viterbi_decoding(tag_scores,self.tag_transitions,self.vt.w2i, self.ntags)

        return [self.vb.i2w[b] for b in bios],[pos_tagger.vt.i2w[t] if pos_tagger else  self.vt.i2w[t] for t in pos_tags]

    def train(self, train_data):
        tagged, loss = 0,0
        best_dev = float('-inf')
        print 'train pos tagger started'
        pos_tagger = self.train_pos_tagger(train_data)
        print 'train pos tagger finished'
        for ITER in xrange(self.options.epochs):
            print 'ITER', ITER
            random.shuffle(train_data)
            batch = []
            for i, s in enumerate(train_data, 1):
                if i % 1000 == 0:
                    self.trainer.status()
                    print loss / tagged
                    loss = 0
                    tagged = 0
                    best_dev = self.validate(best_dev, pos_tagger)
                ws = [self.vw.w2i.get(w, self.UNK_W) for w, p, bio in s]
                ps = [self.vt.w2i[t] for w, t, bio in s]
                bs = [self.vb.w2i[bio] for w, p, bio in s]
                batch.append((ws,ps,bs))
                tagged += len(ps)

                if len(batch)>=self.batch:
                    for j in xrange(len(batch)):
                        ws,_,bs = batch[j]
                        tag_scores = pos_tagger.get_tag_scores(False, words, ws)
                        pos_tags, _ = pos_tagger.viterbi_decoding(tag_scores, pos_tagger.tag_transitions,pos_tagger.vt.w2i, pos_tagger.ntags)
                        sum_errs = self.neg_log_loss([w for w,_,_ in s], ws,  bs, pos_tags)
                        loss += sum_errs.scalar_value()
                    sum_errs.backward()
                    self.trainer.update()
                    renew_cg()
                    batch = []
            self.trainer.status()
            print loss / tagged
            best_dev = self.validate(best_dev, pos_tagger)
        if not options.save_best or not options.dev_file:
            print 'Saving the final model'
            self.save(os.path.join(options.output, options.model))

    def train_pos_tagger(self, train_data):
        pos_train_data = [[(w,p) for w,p,bio in s] for s in train_data]
        pos_dev_data = [[(w,p) for w,p,bio in s] for s in  list(self.read(self.options.dev_file))]
        pos_tagger = Tagger(self.options, self.vw, self.vt, self.vc)
        pos_tagger.train(pos_train_data, pos_dev_data, self.options.pos_epochs, os.path.join(self.options.output, self.options.model+'.pos'))
        return pos_tagger

    def validate(self, best_dev, pos_tagger = None):
        dev = list(self.read(options.dev_file))
        good = bad = 0.0
        good_pos = bad_pos = 0.0
        for sent in dev:
            gold_bios = [b for w, t, b in sent]
            gold_pos = [t for w, t, b in sent]
            bio_tags, pos_tags = [None]*len(gold_bios), [None]*len(gold_pos)
            if pos_tagger:
                bio_tags, pos_tags = self.tag_sent(sent, pos_tagger)
            else:
                words = [w for w, p, bio in sent]
                ws = [self.vw.w2i.get(w, self.UNK_W) for w, p, bio in sent]
                tag_scores = self.get_tag_scores(False, words, ws)
                pt, _ = self.viterbi_decoding(tag_scores, self.tag_transitions, self.vt.w2i,self.ntags)
                pos_tags = [self.vt.i2w[t] for t in pt]

            for go, gp, gu, pp in zip(gold_bios, gold_pos, bio_tags, pos_tags):
                if go == gu:
                    good += 1
                else:
                    bad += 1

                if gp == pp:
                    good_pos += 1
                else:
                    bad_pos += 1
        res = good / (good + bad)
        pos_res = good_pos / (good_pos + bad_pos)
        if res > best_dev:
            print 'dev accuracy (saving):', res, 'pos accuracy', pos_res
            best_dev = res
            self.save(os.path.join(self.options.output, self.options.model))
        else:
            print 'dev accuracy:', res, 'pos accuracy', pos_res
        renew_cg()
        return best_dev

    @staticmethod
    def parse_options():
        parser = OptionParser()
        parser.add_option('--train', dest='conll_train', help='Annotated CONLL train file', metavar='FILE', default='')
        parser.add_option('--dev', dest='dev_file', help='Annotated CONLL development file', metavar='FILE', default=None)
        parser.add_option('--test', dest='conll_test', help='Annotated CONLL test file', metavar='FILE', default='')
        parser.add_option('--inputs', dest='inputs', help='Input tagged files separated by ,', metavar='FILE', default=None)
        parser.add_option('--ext', dest='ext', help='File extension for outputfiles', type='str',default='.chunk')
        parser.add_option('--params', dest='params', help='Parameters file', metavar='FILE', default='params.pickle')
        parser.add_option('--extrn', dest='external_embedding', help='External embeddings', metavar='FILE')
        parser.add_option('--init', dest='initial_embeddings', help='Initial embeddings', metavar='FILE')
        parser.add_option('--model', dest='model', help='Load/Save model file', metavar='FILE', default='model.model')
        parser.add_option('--wembedding', type='int', dest='wembedding_dims', default=128)
        parser.add_option('--cembedding', type='int', dest='cembedding_dims', help='size of character embeddings', default=30)
        parser.add_option('--pembedding', type='int', dest='pembedding_dims', default=30)
        parser.add_option('--epochs', type='int', dest='epochs', default=10)
        parser.add_option('--pos_epochs', type='int', dest='pos_epochs', default=3)
        parser.add_option('--hidden', type='int', dest='hidden_units', default=200)
        parser.add_option('--hidden2', type='int', dest='hidden2_units', default=0)
        parser.add_option('--lstmdims', type='int', dest='lstm_dims', default=200)
        parser.add_option('--tlstmdims', type='int', dest='tag_lstm_dims', default=200)
        parser.add_option('--clstmdims', type='int', dest='clstm_dims', default=100)
        parser.add_option('--outdir', type='string', dest='output', default='')
        parser.add_option('--outfile', type='string', dest='outfile', default='')
        parser.add_option("--eval", action="store_true", dest="eval_format", default=False)
        parser.add_option("--activation", type="string", dest="activation", default="tanh")
        parser.add_option("--drop", action="store_true", dest="drop", default=False, help='Use dropout.')
        parser.add_option("--gru", action="store_true", dest="gru", default=False, help='Use GRU instead of LSTM.')
        parser.add_option("--save_best", action="store_true", dest="save_best", default=False, help='Store the best model.')
        parser.add_option("--dropout", type="float", dest="dropout", default=0.33, help='Dropout probability.')
        parser.add_option('--mem', type='int', dest='mem', default=2048)
        parser.add_option('--k', type='int', dest='k', help = 'word LSTM depth', default=1)
        parser.add_option('--batch', type='int', dest='batch', default=50)
        return parser.parse_args()

if __name__ == '__main__':
    import _dynet as dy
    (options, args) = Chunker.parse_options()
    dyparams = dy.DynetParams()
    dyparams.from_args()
    dyparams.set_mem(options.mem)
    dyparams.init()
    from dynet import *

    if options.conll_train != '' and options.output != '':
        if not os.path.isdir(options.output): os.mkdir(options.output)
        train = list(Chunker.read(options.conll_train))
        print 'load #sent:',len(train)
        words = []
        tags = []
        bios = []
        chars = {' ','<s>','</s>'}
        wc = Counter()
        for s in train:
            for w, p, bio in s:
                words.append(w)
                tags.append(p)
                bios.append(bio)
                [chars.add(x) for x in list(w)]
                wc[w] += 1
        words.append('_UNK_')
        bios.append('_START_')
        bios.append('_STOP_')
        tags.append('_START_')
        tags.append('_STOP_')
        ch = list(chars)

        print 'writing params file'
        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, tags, bios, ch, options), paramsfp)

        vw = util.Vocab.from_corpus([words])
        vb = util.Vocab.from_corpus([bios])
        vt = util.Vocab.from_corpus([tags])
        chars = util.Vocab.from_corpus([ch])

        Chunker(options, vw, vt, chars, vb).train(train)

        options.model = os.path.join(options.output,options.model)
        options.params = os.path.join(options.output,options.params)

    if options.conll_test != '' and options.params != '' and options.model != '' and options.outfile != '':
        print options.model, options.params, options.eval_format
        print 'reading params'
        with open(options.params, 'r') as paramsfp:
            words, tags, bios, ch, opt = pickle.load(paramsfp)
        vw = util.Vocab.from_corpus([words])
        vb = util.Vocab.from_corpus([bios])
        vt = util.Vocab.from_corpus([tags])
        chars = util.Vocab.from_corpus([ch])
        chunker = Chunker(options, vw, vt, chars, vb)
        pos_tagger = Chunker(options, vw, vt, chars, vb)
        print 'loading model'
        print options.model
        chunker.load(options.model)
        pos_tagger.load(options.model+'.pos')

        test = list(Chunker.read(options.conll_test))
        print 'loaded',len(test),'sentences!'
        writer = codecs.open(options.outfile, 'w')
        for sent in test:
            output = list()
            tags, pos_tags = chunker.tag_sent(sent, pos_tagger)
            if options.eval_format:
                 [output.append(' '.join([sent[i][0], pos_tags[i], sent[i][2], tags[i]])) for i in xrange(len(tags))]
            else:
                [output.append(' '.join([sent[i][0], pos_tags[i], tags[i]])) for i in xrange(len(tags))]
            writer.write('\n'.join(output))
            writer.write('\n\n')
        print 'done!'

    if options.inputs != None and options.params != '' and options.model != '':
        print options.model, options.params, options.eval_format
        print 'reading params'
        with open(options.params, 'r') as paramsfp:
            words, tags, bios, ch, opt = pickle.load(paramsfp)
        vw = util.Vocab.from_corpus([words])
        vb = util.Vocab.from_corpus([bios])
        vt = util.Vocab.from_corpus([tags])
        chars = util.Vocab.from_corpus([ch])
        chunker = Chunker(options, vw, vt, chars, vb)
        pos_tagger = Chunker(options, vw, vt, chars, vb)
        print 'loading model'
        print options.model
        chunker.load(options.model)
        pos_tagger.load(options.model+'.pos')

        inputs = options.inputs.strip().split(',')
        for input in inputs:
            print input
            test = list(Chunker.read_raw_file(input))
            print 'loaded',len(test),'sentences!'
            writer = codecs.open(input+options.ext, 'w')
            for sent in test:
                output = list()
                tags, pos_tags = chunker.tag_sent(sent, pos_tagger)
                [output.append(' '.join([sent[i], pos_tags[i], tags[i]])) for i in xrange(len(tags))]
                writer.write('\n'.join(output))
                writer.write('\n\n')
        print 'done!'