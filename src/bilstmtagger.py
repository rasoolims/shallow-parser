from postagger import  *

class Chunker(Tagger):
    def __init__(self, options, words, tags, labels, chars, pos_tagger):
        Tagger.__init__(self, options, words, tags, chars)
        self.pos_tagger = pos_tagger
        if options.tag_init: self.init_pos_tagger()
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify}
        self.activation = self.activations[options.activation]
        self.vl = util.Vocab.from_corpus([labels]) # no-bio tags
        self.nLabels = self.vl.size()
        print  'num of labels',self.nLabels
        self.PE = self.model.add_lookup_parameters((self.ntags, options.pembedding_dims))
        self.H1 = self.model.add_parameters((options.hidden_units, options.lstm_dims)) if options.hidden_units > 0 else None
        self.H2 = self.model.add_parameters((options.hidden2_units, options.hidden_units)) if options.hidden2_units > 0 else None
        hdim = options.hidden2_units if options.hidden2_units>0 else options.hidden_units if options.hidden_units>0 else options.lstm_dims
        self.O = self.model.add_parameters((self.nLabels, hdim))
        self.transitions = self.model.add_lookup_parameters((self.nLabels, self.nLabels))
        inp_dim = options.wembedding_dims + self.edim + options.clstm_dims + self.ntags + options.pembedding_dims
        self.chunk_lstms = BiRNNBuilder(self.k, inp_dim, options.lstm_dims, self.model, LSTMBuilder if not options.gru else GRUBuilder)

    def init_pos_tagger(self):
        for i in xrange(self.nwords):
            self.WE.init_row(i, self.pos_tagger.WE[i].npvalue())
        for i in xrange(self.chars.size()):
            self.CE.init_row(i, self.pos_tagger.CE[i].npvalue())

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

    def forward_semi(self, observations, ntags, trans_matrix, dct):
        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = pick(scores, argmax_score)
            max_score_expr_broadcast = concatenate([max_score_expr] * ntags)
            return max_score_expr + log(sum_cols(transpose(exp(scores - max_score_expr_broadcast))))

        init_alphas = [-1e10] * ntags
        init_alphas[dct['_START_']] = 0
        for_expr = inputVector(init_alphas)

        for i in xrange(len(observations)):
            alphas_t = [scalarInput(0)]*ntags
            for k in xrange(i+1):
                feat = observations[k] - observations[i-1] if i>=0 else observations[k]
                for next_tag in range(ntags):
                    obs_broadcast = concatenate([pick(feat, next_tag)] * ntags)
                    next_tag_expr = for_expr + trans_matrix[next_tag] + obs_broadcast
                    alphas_t[next_tag] = alphas_t[next_tag]+ log_sum_exp(next_tag_expr)
            for_expr = concatenate(alphas_t)
        terminal_expr = for_expr + trans_matrix[dct['_STOP_']]
        alpha = log_sum_exp(terminal_expr)
        return alpha

    def get_chunk_lstm_features(self, is_train, sent_words, words,auto_tags):
        tag_lstm, char_lstms, wembs, evec = self.get_pos_lstm_features(is_train, sent_words, words)
        pembs = [noise(self.PE[p], 0.001) if is_train else self.PE[p] for p in auto_tags]
        O = parameter(self.tagO)
        tag_scores = []
        for f in tag_lstm:
            score_t = O * f
            tag_scores.append(score_t)
        inputs = [concatenate(filter(None, [wembs[i], evec[i], char_lstms[i][-1], softmax(tag_scores[i]), pembs[i]])) for i in xrange(len(words))]
        input_lstm = self.chunk_lstms.transduce(inputs)
        return input_lstm

    def score_sentence_semi(self, observations, segments, trans_matrix, dct):
        score_seq = [0]
        score = scalarInput(0)
        labels = [dct['_START_']] + [dct[l] for _,_,l in segments]

        for i in xrange(len(segments)):
            s,e,_ = segments[i]
            score = score + pick(trans_matrix[labels[i+1]],labels[i]) + pick(observations[e], labels[i+1]) - (pick(observations[s-1], labels[i]) if s>0 else scalarInput(0))
            score_seq.append(score.value())
        score = score + pick(trans_matrix[dct['_STOP_']],labels[-1])
        return score

    def viterbi_decoding_semi(self, observations, trans_matrix, dct, nL):
        backpointers = []
        init_vvars   = [-1e10] * nL
        init_vvars[dct['_START_']] = 0 # <Start> has all the probability
        for_expr = [inputVector(0)]*(len(observations)+1)
        for_expr[0] = inputVector(init_vvars)
        trans_exprs  = [trans_matrix[idx] for idx in range(nL)]
        for i in xrange(len(observations)):
            bptrs_t = []
            vvars_t = []
            for next_tag in range(nL):
                max_pointer = 0
                max_value = float('-inf')
                for k in xrange(i+1):
                    next_tag_expr = for_expr[k] + trans_exprs[next_tag] + observations[i] - (observations[k-1] if k>0 else inputVector(0))
                    next_tag_arr = next_tag_expr.npvalue()
                    best_tag_id  = np.argmax(next_tag_arr)
                    v = pick(next_tag_expr, best_tag_id).value()
                    if v>max_value:
                        max_value = v
                        max_pointer =best_tag_id,k,pick(next_tag_expr, best_tag_id)
                bptrs_t.append((max_pointer[0],max_pointer[1]))
                vvars_t.append(max_pointer[2])
            for_expr[i] = concatenate(vvars_t)
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr[-1] + trans_exprs[dct['_STOP_']]
        terminal_arr  = terminal_expr.npvalue()
        best_tag_id = np.argmax(terminal_arr)
        path_score  = pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        segments = []
        end = len(observations)-1
        for bptrs_t in reversed(backpointers):
            best_tag_id,start = bptrs_t[best_tag_id]
            segments.append((start,end,best_tag_id))
            end = start-1
        #start = segments.pop() # Remove the start symbol
        segments.reverse()

        bios = []
        for s,e,l in segments:
            if l == 'O':
                bios.append(l)
            else:
                bios.append('B-'+l)
            for i in range(s,e):
                if l == 'O':
                    bios.append(l)
                else:
                    bios.append('I-' + l)

        assert start == dct['_START_']
        # Return best path and best path's score
        return bios, path_score

    def build_graph(self, sent_words, words, auto_tags, is_train):
        input_lstm = self.get_chunk_lstm_features(is_train, sent_words, words, auto_tags)
        H1 = parameter(self.H1) if self.H1 != None else None
        H2 = parameter(self.H2) if self.H2 != None else None
        O = parameter(self.O)
        scores = []

        for f in input_lstm:
            score_t = O*(self.activation(H2*self.activation(H1 * f))) if H2!=None else O * (self.activation(H1 * f)) if self.H1 != None  else O * f
            scores.append(score_t)
        return scores

    def neg_log_loss(self, sent_words, words, segments, auto_tags):
        observations = self.build_graph(sent_words, words, auto_tags, True)
        gold_score = self.score_sentence_semi(observations, segments, self.transitions, self.vl.w2i)
        forward_score = self.forward_semi(observations, self.nLabels, self.transitions, self.vl.w2i)
        return forward_score - gold_score

    def tag_sent(self, sent):
        renew_cg()
        words = [w for w, p, bio in sent]
        ws = [self.vw.w2i.get(w, self.UNK_W) for w, p, bio in sent]
        auto_tags = self.pos_tagger.best_pos_tags(words)
        observations = self.build_graph(words, ws, auto_tags, False)
        bios, score = self.viterbi_decoding_semi(observations,self.transitions,self.vl.w2i, self.nLabels)
        return bios,[self.pos_tagger.vt.i2w[p] for p in auto_tags]

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
                    if ITER >= self.options.pos_epochs: best_dev = self.validate(best_dev)
                ws = [self.vw.w2i.get(w, self.UNK_W) for w, p, bio in s]
                ps = [self.vt.w2i[t] for w, t, bio in s]
                auto_tags = self.pos_tagger.best_pos_tags([w for w, p, bio in s])

                segments = self.get_segments(s)
                batch.append(([w for w,_,_ in s],ws,ps,segments,auto_tags))
                tagged += len(ps)

                if len(batch)>=self.batch:
                    errs = []
                    for j in xrange(len(batch)):
                        sent_words,ws,ps,segments,at = batch[j]
                        if ITER < self.options.pos_epochs:
                            errs.append(self.pos_neg_log_loss(sent_words, ws,  ps))
                        else:
                            errs.append(self.neg_log_loss(sent_words, ws,  segments, at))
                    sum_errs = esum(errs)
                    loss += sum_errs.scalar_value()
                    sum_errs.backward()
                    self.trainer.update()
                    renew_cg()
                    batch = []
            self.trainer.status()
            print loss / tagged
            if ITER >= self.options.pos_epochs: best_dev = self.validate(best_dev)
        if not options.save_best or not options.dev_file:
            print 'Saving the final model'
            self.save(os.path.join(options.output, options.model))

    def get_segments(self, sen):
        bios = [bio for _, _, bio in sen]
        segments = []
        s, e, l = 0, 0, ''
        r = False
        for i in xrange(len(bios)):
            if bios[i] == 'O':
                if r: segments.append((s, e, bios[i - 1][bios[i - 1].find('-') + 1:]))
                segments.append((i, i, bios[i]))
                r = False
                s,e = i + 1,i + 1
            elif bios[i].startswith('B-') and not r:
                segments.append((s, e, bios[i - 1][bios[i - 1].find('-') + 1:]))
                s,e = i,i
            else:
                e = i
                r = True
        if r: segments.append((s, e, bios[-1][bios[-1].find('-') + 1:]))
        return segments

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
        tags = []
        labels = []
        chars = {' ','<s>','</s>'}
        wc = Counter()
        for s in train:
            for w, p, bio in s:
                words.append(w)
                tags.append(p)
                labels.append(bio[bio.find('-')+1:] if '-' in bio else bio)
                [chars.add(x) for x in list(w)]
                wc[w] += 1
        words.append('_UNK_')
        labels.append('_START_')
        labels.append('_STOP_')
        tags.append('_START_')
        tags.append('_STOP_')
        ch = list(chars)

        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, tags, labels, ch, options), paramsfp)

        Chunker(options, words, tags, labels, ch, tagger).train()

        options.model = os.path.join(options.output,options.model)
        options.params = os.path.join(options.output,options.params)

    if options.conll_test != '' and options.params != '' and options.model != '' and options.outfile != '':
        print options.model, options.params, options.eval_format
        print 'reading params'
        with open(options.params, 'r') as paramsfp:
            words, tags, labels, ch, opt = pickle.load(paramsfp)
        chunker = Chunker(opt, words, tags, labels, ch, tagger)

        print 'loading model'
        print options.model
        chunker.load(options.model)

        test = list(Chunker.read(options.conll_test))
        print 'loaded',len(test),'sentences!'
        writer = codecs.open(options.outfile, 'w')
        for sent in test:
            output = list()
            tags, pos_tags = chunker.tag_sent(sent)
            if options.eval_format:
                 [output.append(' '.join([sent[i][0], pos_tags[i], sent[i][2], tags[i]])) for i in xrange(len(tags))]
            else:
                [output.append(' '.join([sent[i][0], pos_tags[i], tags[i]])) for i in xrange(len(tags))]
            writer.write('\n'.join(output))
            writer.write('\n\n')
        print 'done!'