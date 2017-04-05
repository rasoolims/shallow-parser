from collections import Counter
import random,os,codecs,pickle,time
from optparse import OptionParser
import numpy as np

import util

class Tagger:
    def __init__(self, options, words, tags, bios, chars):
        self.options = options
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify}
        self.activation = self.activations[options.activation]
        self.vw = util.Vocab.from_corpus([words])
        self.vb = util.Vocab.from_corpus([bios])
        self.vt = util.Vocab.from_corpus([tags])
        self.UNK_W = self.vw.w2i['_UNK_']
        self.batch = options.batch
        self.nwords = self.vw.size()
        self.chars = util.Vocab.from_corpus([chars])
        self.ntags = self.vt.size()
        self.nBios = self.vb.size()
        print 'num of pos tags',self.ntags, 'num of bio tags',self.nBios
        self.model = Model()
        self.trainer = AdamTrainer(self.model)

        self.WE = self.model.add_lookup_parameters((self.nwords, options.wembedding_dims))
        self.CE = self.model.add_lookup_parameters((self.chars.size(), options.cembedding_dims))
        self.pH1 = self.model.add_parameters((options.hidden_units, options.lstm_dims)) if options.hidden_units > 0 else None
        self.pH2 = self.model.add_parameters((options.hidden2_units, options.hidden_units)) if options.hidden2_units > 0 else None
        hdim = options.hidden2_units if options.hidden2_units>0 else options.hidden_units if options.hidden_units>0 else options.lstm_dims
        self.pO = self.model.add_parameters((self.nBios, hdim))
        self.k = options.k
        self.drop = options.drop
        self.dropout = options.dropout
        self.transitions = self.model.add_lookup_parameters((self.nBios, self.nBios))
        self.tag_transitions = self.model.add_lookup_parameters((self.ntags, self.ntags))
        self.edim = 0
        self.external_embedding = None
        if options.initial_embeddings is not None:
            initial_embeddings_fp = open(options.initial_embeddings, 'r')
            initial_embeddings_fp.readline()
            initial_embeddings_vec = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                                       initial_embeddings_fp}
            initial_embeddings_fp.close()
            for word in self.vw.w2i.keys():
               if word in initial_embeddings_vec:
                   assert options.wembedding_dims == len(initial_embeddings_vec[word])
                   self.WE.init_row(self.vw.w2i.get(word), initial_embeddings_vec[word])
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
            self.extrnd['_UNK_'] = 1
            self.extrnd['_START_'] = 2
            self.extrn_lookup.init_row(1, noextrn)
            print 'Loaded external embedding. Vector dimensions:', self.edim

        tag_inp_dim = options.wembedding_dims + self.edim + options.clstm_dims
        inp_dim = tag_inp_dim + self.ntags
        self.chunk_lstms = BiRNNBuilder(self.k, inp_dim, options.lstm_dims, self.model, LSTMBuilder if not options.gru else GRUBuilder)
        self.tag_lstms = BiRNNBuilder(self.k, tag_inp_dim, options.tag_lstm_dims, self.model, LSTMBuilder if not options.gru else GRUBuilder)
        self.char_lstms = BiRNNBuilder(1, options.cembedding_dims, options.clstm_dims, self.model, LSTMBuilder if not options.gru else GRUBuilder)
        self.tagO = self.model.add_parameters((self.ntags, options.tag_lstm_dims))

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
    def read_tagged_file(fname):
        for line in file(fname):
            spl = line.strip().split()
            sent = []
            for s in spl:
                w = s[:s.rfind('_')]
                p = s[s.rfind('_')+1:]
                sent.append((w,p,'_'))
            yield sent

    def build_pos_graph(self, sent_words, words, is_train):
        input_lstm = self.get_lstm_features(is_train, sent_words, words, False)[0]

        O = parameter(self.tagO)
        probs = []
        for f in input_lstm:
            score_t = O*f
            probs.append(score_t)
        return probs

    def get_lstm_features(self, is_train, sent_words, words, is_chunking):
        if is_chunking:
            tag_lstm, char_lstms, wembs, evec = self.get_lstm_features(is_train, sent_words, words, False)
            O = parameter(self.tagO)
            tag_scores = []
            for f in tag_lstm:
                score_t = O * f
                tag_scores.append(score_t)
            inputs = [concatenate(filter(None, [wembs[i], evec[i], char_lstms[i][-1], softmax(tag_scores[i])])) for i in xrange(len(words))]
            input_lstm = self.chunk_lstms.transduce(inputs)
            return input_lstm,tag_scores
        else:
            char_lstms = []
            for w in sent_words:
                char_lstms.append(self.char_lstms.transduce([self.CE[self.chars.w2i[c]] if (c in self.chars.w2i and not is_train) or (is_train and random.random() >= 0.001) else self.CE[self.chars.w2i[' ']] for c in ['<s>'] + list(w) + ['</s>']]))
            wembs = [noise(self.WE[w], 0.1) if is_train else self.WE[w] for w in words]
            evec = [self.extrn_lookup[self.extrnd[w]] if self.edim > 0 and w in self.extrnd else self.extrn_lookup[
                1] if self.edim > 0 else None for w in words]
            inputs = [concatenate(filter(None, [wembs[i], evec[i], char_lstms[i][-1]])) for i in xrange(len(words))]
            if self.drop:
                [dropout(inputs[i], self.dropout) for i in xrange(len(inputs))]
            input_lstm = self.tag_lstms.transduce(inputs)
            return input_lstm, char_lstms, wembs, evec

    def pos_loss(self, sent_words, words, tags):
        probs = self.build_pos_graph(sent_words, words, True)
        errs = []
        for i in xrange(len(tags)):
            err = -log(pick(probs[i], tags[i]))
            errs.append(err)
        return errs

    def build_tagging_graph(self, sent_words, words, is_train):
        input_lstm,pos_probs = self.get_lstm_features(is_train, sent_words, words, True)
        H1 = parameter(self.pH1) if self.pH1!=None else None
        H2 = parameter(self.pH2) if self.pH2!=None else None
        O = parameter(self.pO)
        scores = []

        for f in input_lstm:
            score_t = O*(self.activation(H2*self.activation(H1 * f))) if H2!=None else O * (self.activation(H1 * f)) if self.pH1!=None  else O*f
            scores.append(score_t)
        return scores,pos_probs

    def score_sentence(self, observations, labels, trans_matrix, dct):
        assert len(observations) == len(labels)
        score_seq = [0]
        score = scalarInput(0)
        labels = [dct['_START_']] + labels
        for i, obs in enumerate(observations):
            score = score + pick(trans_matrix[labels[i+1]],labels[i]) + pick(obs, labels[i+1])
            score_seq.append(score.value())
        score = score + pick(trans_matrix[dct['_STOP_']],labels[-1])
        return score

    def forward(self, observations, ntags, trans_matrix, dct):
        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = pick(scores, argmax_score)
            max_score_expr_broadcast = concatenate([max_score_expr] * ntags)
            return max_score_expr + log(sum_cols(transpose(exp(scores - max_score_expr_broadcast))))

        init_alphas = [-1e10] * ntags
        init_alphas[dct['_START_']] = 0
        for_expr = inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(ntags):
                obs_broadcast = concatenate([pick(obs, next_tag)] * ntags)
                next_tag_expr = for_expr + trans_matrix[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = concatenate(alphas_t)
        terminal_expr = for_expr + trans_matrix[dct['_STOP_']]
        alpha = log_sum_exp(terminal_expr)
        return alpha

    def viterbi_decoding(self, observations, trans_matrix, dct, nL):
        backpointers = []
        init_vvars   = [-1e10] * nL
        init_vvars[dct['_START_']] = 0 # <Start> has all the probability
        for_expr = inputVector(init_vvars)
        trans_exprs  = [trans_matrix[idx] for idx in range(nL)]
        for obs in observations:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(nL):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id  = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(pick(next_tag_expr, best_tag_id))
            for_expr = concatenate(vvars_t) + obs
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[dct['_STOP_']]
        terminal_arr  = terminal_expr.npvalue()
        best_tag_id = np.argmax(terminal_arr)
        path_score  = pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id] # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop() # Remove the start symbol
        best_path.reverse()
        assert start == dct['_START_']
        # Return best path and best path's score
        return best_path, path_score

    def neg_log_loss(self, sent_words, words, labels, is_chunking):
        observations = self.build_tagging_graph(sent_words, words, True)[0] if is_chunking else self.build_pos_graph(sent_words, words, True)
        gold_score = self.score_sentence(observations, labels, self.transitions if is_chunking else self.tag_transitions, self.vb.w2i if is_chunking else self.vt.w2i)
        forward_score = self.forward(observations, self.nBios if is_chunking else self.ntags, self.transitions if is_chunking else self.tag_transitions, self.vb.w2i if is_chunking else self.vt.w2i)
        return forward_score - gold_score

    def viterbi_loss(self,  sent_words,words,tags, bios):
        observations = self.build_tagging_graph(sent_words,words,tags, True)[0]
        viterbi_tags, viterbi_score = self.viterbi_decoding(observations,self.transitions,self.vb.w2i, self.nBios)
        if viterbi_tags != bios:
            gold_score = self.score_sentence(observations, bios, self.transitions, self.vb.w2i)
            return (viterbi_score - gold_score), viterbi_tags
        else:
            return dy.scalarInput(0), viterbi_tags


    def tag_sent(self, sent):
        renew_cg()
        words = [w for w, p, bio in sent]
        ws = [self.vw.w2i.get(w, self.UNK_W) for w, p, bio in sent]
        observations,tag_scores = self.build_tagging_graph(words, ws, False)
        bios, score = self.viterbi_decoding(observations,self.transitions,self.vb.w2i, self.nBios)
        pos_tags, _ = self.viterbi_decoding(tag_scores,self.tag_transitions,self.vt.w2i, self.ntags)
        return [self.vb.i2w[b] for b in bios],[self.vt.i2w[t] for t in pos_tags]

    def train(self):
        tagged, loss = 0,0
        best_dev = float('-inf')

        # First train pos tagger for a while
        for ITER in xrange(self.options.pos_epochs):
            print 'ITER', ITER
            random.shuffle(train)
            batch = []
            for i, s in enumerate(train, 1):
                if i % 1000 == 0:
                    self.trainer.status()
                    print loss / tagged
                    loss = 0
                    tagged = 0
                    self.validate(0, False)
                ws = [self.vw.w2i.get(w, self.UNK_W) for w, p, bio in s]
                ps = [self.vt.w2i[t] for w, t, bio in s]
                bs = [self.vb.w2i[bio] for w, p, bio in s]
                batch.append((ws,ps,bs))
                tagged += len(ps)

                if len(batch)>=self.batch:
                    for j in xrange(len(batch)):
                        ws,ps,_ = batch[j]
                        sum_errs = self.neg_log_loss([w for w,_,_ in s], ws,  ps, False)
                        loss+= sum_errs.scalar_value()
                    sum_errs.backward()
                    self.trainer.update()
                    tagged += len(ps)
                    renew_cg()
                    batch = []
            self.trainer.status()
            self.validate(0, False)

        loss = 0
        tagged = 0
        for ITER in xrange(self.options.epochs):
            print 'ITER', ITER
            random.shuffle(train)
            batch = []
            for i, s in enumerate(train, 1):
                if i % 1000 == 0:
                    self.trainer.status()
                    print loss / tagged
                    loss = 0
                    tagged = 0
                    best_dev = self.validate(best_dev)
                ws = [self.vw.w2i.get(w, self.UNK_W) for w, p, bio in s]
                ps = [self.vt.w2i[t] for w, t, bio in s]
                bs = [self.vb.w2i[bio] for w, p, bio in s]
                batch.append((ws,ps,bs))
                tagged += len(ps)

                if len(batch)>=self.batch:
                    for j in xrange(len(batch)):
                        ws,_,bs = batch[j]
                        sum_errs = self.neg_log_loss([w for w,_,_ in s], ws,  bs, True)
                        loss += sum_errs.scalar_value()
                    sum_errs.backward()
                    self.trainer.update()
                    renew_cg()
                    batch = []
            self.trainer.status()
            best_dev = self.validate(best_dev)
        if not options.save_best or not options.dev_file:
            print 'Saving the final model'
            self.save(os.path.join(options.output, options.model))

    def validate(self, best_dev, save=True):
        dev = list(self.read(options.dev_file))
        good = bad = 0.0
        good_pos = bad_pos = 0.0
        if options.save_best and options.dev_file:
            for sent in dev:
                bio_tags, pos_tags = self.tag_sent(sent)
                gold_bois = [b for w, t, b in sent]
                gold_pos = [t for w, t, b in sent]
                for go, gp, gu, pp in zip(gold_bois, gold_pos, bio_tags, pos_tags):
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
            if save:
                if res > best_dev:
                    print '\ndev accuracy (saving):', res, 'pos accuracy', pos_res
                    best_dev = res
                    self.save(os.path.join(options.output, options.model))
                else:
                    print '\ndev accuracy:', res, 'pos accuracy', pos_res
            else:
                print '\ndev pos accuracy', pos_res
        return best_dev

    def load(self, f):
        self.model.load(f)

    def save(self, f):
        self.model.save(f)

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
        parser.add_option('--epochs', type='int', dest='epochs', default=5)
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
    (options, args) = Tagger.parse_options()
    dyparams = dy.DynetParams()
    dyparams.from_args()
    dyparams.set_mem(options.mem)
    dyparams.init()
    from dynet import *

    if options.conll_train != '' and options.output != '':
        if not os.path.isdir(options.output): os.mkdir(options.output)
        train = list(Tagger.read(options.conll_train))
        print 'load #sent:',len(train)
        words = []
        bio_tags = []
        bios = []
        chars = {' ','<s>','</s>'}
        wc = Counter()
        for s in train:
            for w, p, bio in s:
                words.append(w)
                bio_tags.append(p)
                bios.append(bio)
                [chars.add(x) for x in list(w)]
                wc[w] += 1
        words.append('_UNK_')
        bios.append('_START_')
        bios.append('_STOP_')
        bio_tags.append('_START_')
        bio_tags.append('_STOP_')
        ch = list(chars)

        print 'writing params file'
        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, bio_tags, bios, ch, options), paramsfp)

        Tagger(options, words, bio_tags, bios, ch).train()

        options.model = os.path.join(options.output,options.model)
        options.params = os.path.join(options.output,options.params)

    if options.conll_test != '' and options.params != '' and options.model != '' and options.outfile != '':
        print options.model, options.params, options.eval_format
        print 'reading params'
        with open(options.params, 'r') as paramsfp:
            words, bio_tags, bios, ch, opt = pickle.load(paramsfp)
        tagger = Tagger(opt, words, bio_tags, bios, ch)

        print 'loading model'
        print options.model
        tagger.load(options.model)

        test = list(Tagger.read(options.conll_test))
        print 'loaded',len(test),'sentences!'
        writer = codecs.open(options.outfile, 'w')
        for sent in test:
            output = list()
            bio_tags,pos_tags = tagger.tag_sent(sent)
            if options.eval_format:
                 [output.append(' '.join([sent[i][0], pos_tags[i][1], sent[i][2], bio_tags[i]])) for i in xrange(len(bio_tags))]
            else:
                [output.append(' '.join([sent[i][0], pos_tags[i][1], bio_tags[i]])) for i in xrange(len(bio_tags))]
            writer.write('\n'.join(output))
            writer.write('\n\n')
        print 'done!'

    if options.inputs != None and options.params != '' and options.model != '':
        print options.model, options.params, options.eval_format
        print 'reading params'
        with open(options.params, 'r') as paramsfp:
            words, bio_tags, bios, ch, opt = pickle.load(paramsfp)
        tagger = Tagger(opt, words, bio_tags, bios, ch)
        print 'loading model'
        print options.model
        tagger.load(options.model)

        inputs = options.inputs.strip().split(',')
        for input in inputs:
            print input
            test = list(Tagger.read_tagged_file(input))
            print 'loaded',len(test),'sentences!'
            writer = codecs.open(input+options.ext, 'w')
            for sent in test:
                output = list()
                bio_tags,pos_tags = tagger.tag_sent(sent)
                if options.eval_format:
                     [output.append(' '.join([sent[i][0], pos_tags[i][1], sent[i][2], bio_tags[i]])) for i in xrange(len(bio_tags))]
                else:
                    [output.append(' '.join([sent[i][0], pos_tags[i][1], bio_tags[i]])) for i in xrange(len(bio_tags))]
                writer.write('\n'.join(output))
                writer.write('\n\n')
        print 'done!'