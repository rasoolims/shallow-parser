from postagger import  *

class Chunker(Tagger):
    def __init__(self, options, words, tags, bios, chars, pos_tagger):
        Tagger.__init__(self, options, words, tags, chars)
        self.pos_tagger = pos_tagger
        if options.tag_init: self.init_pos_tagger()
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify}
        self.activation = self.activations[options.activation]
        self.vb = util.Vocab.from_corpus([bios])
        self.nBios = self.vb.size()
        print  'num of bio tags',self.nBios

        self.H1 = self.model.add_parameters((options.hidden_units, options.lstm_dims)) if options.hidden_units > 0 else None
        self.H2 = self.model.add_parameters((options.hidden2_units, options.hidden_units)) if options.hidden2_units > 0 else None
        hdim = options.hidden2_units if options.hidden2_units>0 else options.hidden_units if options.hidden_units>0 else options.lstm_dims
        self.O = self.model.add_parameters((self.nBios, hdim))
        self.transitions = self.model.add_lookup_parameters((self.nBios, self.nBios))

        inp_dim = options.wembedding_dims + self.edim + options.clstm_dims + self.ntags
        self.chunk_lstms = BiRNNBuilder(self.k, inp_dim, options.lstm_dims, self.model, LSTMBuilder if not options.gru else GRUBuilder)

    def init_pos_tagger(self):
        for i in xrange(self.nwords):
            self.WE.init_row(i, self.pos_tagger.WE[i].npvalue())
        for i in xrange(self.chars.size()):
            self.CE.init_row(i, self.pos_tagger.CE[i].npvalue())
        #todo try to initialize layers as well

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
        H1 = parameter(self.H1) if self.H1 != None else None
        H2 = parameter(self.H2) if self.H2 != None else None
        O = parameter(self.O)
        scores = []

        for f in input_lstm:
            score_t = O*(self.activation(H2*self.activation(H1 * f))) if H2!=None else O * (self.activation(H1 * f)) if self.H1 != None  else O * f
            scores.append(score_t)
        return scores,pos_probs

    def neg_log_loss(self, sent_words, words, labels, is_chunking):
        observations = self.build_tagging_graph(sent_words, words, True)[0] if is_chunking else self.build_pos_graph(sent_words, words, True)
        gold_score = self.score_sentence(observations, labels, self.transitions if is_chunking else self.tag_transitions, self.vb.w2i if is_chunking else self.vt.w2i)
        forward_score = self.forward(observations, self.nBios if is_chunking else self.ntags, self.transitions if is_chunking else self.tag_transitions, self.vb.w2i if is_chunking else self.vt.w2i)
        return forward_score - gold_score

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
            print loss / tagged
            best_dev = self.validate(best_dev)
        if not options.save_best or not options.dev_file:
            print 'Saving the final model'
            self.save(os.path.join(options.output, options.model))

    def validate(self, best_dev):
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
            if res > best_dev:
                print 'dev accuracy (saving):', res, 'pos accuracy', pos_res
                best_dev = res
                self.save(os.path.join(options.output, options.model))
            else:
                print 'dev accuracy:', res, 'pos accuracy', pos_res
        return best_dev


if __name__ == '__main__':
    print 'reading pos tagger'
    with open(options.pos_params, 'r') as paramsfp:
        p_words, p_tags, p_ch, p_opt = pickle.load(paramsfp)
    tagger = Tagger(p_opt, p_words, p_tags, p_ch)
    tagger.load(options.pos_model)
    print 'writing params file'

    if options.conll_train != '' and options.output != '':
        if not os.path.isdir(options.output): os.mkdir(options.output)
        train = list(Chunker.read(options.conll_train))
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

        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, bio_tags, bios, ch, options), paramsfp)

        Chunker(options, words, bio_tags, bios, ch,tagger).train()

        options.model = os.path.join(options.output,options.model)
        options.params = os.path.join(options.output,options.params)

    if options.conll_test != '' and options.params != '' and options.model != '' and options.outfile != '':
        print options.model, options.params, options.eval_format
        print 'reading params'
        with open(options.params, 'r') as paramsfp:
            words, bio_tags, bios, ch, opt = pickle.load(paramsfp)
        chunker = Chunker(opt, words, bio_tags, bios, ch,tagger)

        print 'loading model'
        print options.model
        chunker.load(options.model)

        test = list(Chunker.read(options.conll_test))
        print 'loaded',len(test),'sentences!'
        writer = codecs.open(options.outfile, 'w')
        for sent in test:
            output = list()
            bio_tags,pos_tags = chunker.tag_sent(sent)
            if options.eval_format:
                 [output.append(' '.join([sent[i][0], pos_tags[i], sent[i][2], bio_tags[i]])) for i in xrange(len(bio_tags))]
            else:
                [output.append(' '.join([sent[i][0], pos_tags[i], bio_tags[i]])) for i in xrange(len(bio_tags))]
            writer.write('\n'.join(output))
            writer.write('\n\n')
        print 'done!'