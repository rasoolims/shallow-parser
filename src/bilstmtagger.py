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

    def forward_semi(self, observations, ntags, trans_matrix, dct,longest):
        def log_sum_exp(scores,ln):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = pick(scores, argmax_score)
            max_score_expr_broadcast = concatenate([max_score_expr] * ln)
            return max_score_expr + log(sum_cols(transpose(exp(scores - max_score_expr_broadcast))))

        init_alphas = [-1e10] * ntags
        init_alphas[dct['_START_']] = 0
        for_expr = [inputVector([0] * ntags)] * (len(observations) + 1)
        for_expr[0] = inputVector(init_alphas)

        for i in xrange(len(observations)):
            alphas_t = []
            for next_tag in range(ntags):
                a = []
                for k in xrange(i+1):
                    a.append(for_expr[k] + trans_matrix[next_tag] + concatenate([pick(observations[i][k], next_tag)] * ntags))
                alphas_t.append(log_sum_exp(concatenate([a[j] for j in xrange(i+1)]),(i+1)*ntags))
            for_expr[i+1] = concatenate(alphas_t)
        terminal_expr = for_expr[-1] + trans_matrix[dct['_STOP_']]
        alpha = log_sum_exp(terminal_expr, ntags)
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
            score = score + pick(trans_matrix[labels[i+1]],labels[i]) + pick(observations[e][s], labels[i+1])
            score_seq.append(score.value())
        score = score + pick(trans_matrix[dct['_STOP_']],labels[-1])
        return score

    def next_action(self, observations, segments, score, index):
        best_score,best_end,best_label,best_sc = float('-inf'),index,0,None
        prev_label = self.vl.w2i['_START_'] if index == 0 else segments[-1][2]
        for j in range(index, len(observations)):
            for label in xrange(self.nLabels):
                sc = score + pick(self.transitions[label], prev_label) + pick(observations[j][index], label)
                if sc.value()>best_score:
                    best_score = sc
                    best_label = label
                    best_end = j
                    best_sc = sc
        score = best_sc
        if best_end==len(observations)-1:
            score = score + pick(self.transitions[self.vl.w2i['_STOP_']],best_label)
        segments.append((index, best_end, best_label))
        return score,best_end+1

    def greedy_tag(self, observations):
        index,segments,score = 0,[],scalarInput(0)
        while index<len(observations)-1:
            score,index = self.next_action(observations, segments, score, index)
        return segments,score

    def tag_greedy_search(self,segments):
        bios = []
        for s, e, l in segments:
            label = self.vl.i2w[l]
            if label == 'O':
                bios.append(label)
            else:
                bios.append('B-' + label)
            for i in range(s, e):
                if label == 'O':
                    bios.append(label)
                else:
                    bios.append('I-' + label)
        return bios

    def margin_loss(self, sent_words, words, segments, longest, auto_tags):
        observations = self.build_graph(sent_words, words, auto_tags, True)
        gold_score = self.score_sentence_semi(observations, segments, self.transitions, self.vl.w2i)
        _,best_score = self.greedy_tag(observations)
        if (gold_score - best_score).value() < 1.0:
            return  scalarInput(1) + best_score - gold_score
        else:
            return scalarInput(0)

    def viterbi_decoding_semi(self, observations, trans_matrix, dct, nL):
        backpointers = []
        init_vvars   = [-1e10] * nL
        init_vvars[dct.w2i['_START_']] = 0 # <Start> has all the probability
        init_vec = [0] * nL
        for_expr = [inputVector(init_vec)]*(len(observations)+1)
        for_expr[0] = inputVector(init_vvars)
        trans_exprs  = [trans_matrix[idx] for idx in range(nL)]
        for i in xrange(len(observations)):
            bptrs_t = []
            vvars_t = []
            for next_tag in range(nL):
                max_pointer = 0
                max_value = float('-inf')
                for k in xrange(i+1):
                    next_tag_expr = for_expr[k] + trans_exprs[next_tag] + observations[i][k]
                    next_tag_arr = next_tag_expr.npvalue()
                    best_tag_id  = np.argmax(next_tag_arr)
                    v = pick(next_tag_expr, best_tag_id).value()
                    if v>max_value:
                        max_value = v
                        max_pointer =best_tag_id,k,pick(next_tag_expr, best_tag_id)
                bptrs_t.append((max_pointer[0],max_pointer[1]))
                vvars_t.append(max_pointer[2])
            for_expr[i+1] = concatenate(vvars_t)
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr[-1] + trans_exprs[dct.w2i['_STOP_']]
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
            label = dct.i2w[l]
            if label == 'O':
                bios.append(label)
            else:
                bios.append('B-'+label)
            for i in range(s,e):
                if label == 'O':
                    bios.append(label)
                else:
                    bios.append('I-' + label)

        # Return best path and best path's score
        return bios, path_score

    def build_graph(self, sent_words, words, auto_tags, is_train):
        input_lstm = self.get_chunk_lstm_features(is_train, sent_words, words, auto_tags)
        H1 = parameter(self.H1) if self.H1 != None else None
        H2 = parameter(self.H2) if self.H2 != None else None
        O = parameter(self.O)
        scores = [None]*len(input_lstm)

        for i in xrange(len(input_lstm)):
            scores[i] = [None]*(i+1)
            for k in xrange(i+1):
                f = input_lstm[i] - input_lstm[k-1] if k>0 else input_lstm[i]
                score_t = O * (self.activation(H2 * self.activation(H1 * f))) if H2 != None else O * (self.activation(H1 * f)) if self.H1 != None  else O * f
                scores[i][k] = score_t
        return scores

    def neg_log_loss(self, sent_words, words, segments, longest, auto_tags):
        observations = self.build_graph(sent_words, words, auto_tags, True)
        gold_score = self.score_sentence_semi(observations, segments, self.transitions, self.vl.w2i)
        gold_score = self.score_sentence_semi(observations, segments, self.transitions, self.vl.w2i)
        forward_score = self.forward_semi(observations, self.nLabels, self.transitions, self.vl.w2i, longest)
        #assert (forward_score - log(gold_score)).value()>=0
        return forward_score - gold_score

    def tag_sent(self, sent):
        renew_cg()
        words = [w for w, p, bio in sent]
        ws = [self.vw.w2i.get(w, self.UNK_W) for w, p, bio in sent]
        auto_tags = self.pos_tagger.best_pos_tags(words)
        observations = self.build_graph(words, ws, auto_tags, False)
        bios = self.tag_greedy_search(self.greedy_tag(observations)[0])
        return bios,[self.pos_tagger.vt.i2w[p] for p in auto_tags]

    def chunk_raw_sent(self, sent):
        renew_cg()
        ws = [self.vw.w2i.get(w, self.UNK_W) for w in sent]
        auto_tags = self.pos_tagger.best_pos_tags(sent)
        observations = self.build_graph(sent, ws, auto_tags, False)
        bios, score = self.viterbi_decoding(observations,self.transitions,self.vb.w2i, self.nBios)
        return [self.vb.i2w[b] for b in bios],[self.pos_tagger.vt.i2w[p] for p in auto_tags]

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

                segments,l = self.get_segments(s)
                batch.append(([w for w,_,_ in s],ws,ps,segments,l,auto_tags))
                tagged += len(ps)

                if len(batch)>=self.batch:
                    errs = []
                    for j in xrange(len(batch)):
                        sent_words,ws,ps,segments,l,at = batch[j]
                        if ITER < self.options.pos_epochs:
                            errs.append(self.pos_neg_log_loss(sent_words, ws,  ps))
                        else:
                            errs.append(self.margin_loss(sent_words, ws,  segments, l,at))
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
        longest = 1
        for i in xrange(len(bios)):
            if bios[i] == 'O':
                if r:
                    segments.append((s, e, bios[i - 1][bios[i - 1].find('-') + 1:]))
                    if e-s+1> longest: longest = e-s+1
                segments.append((i, i, bios[i]))
                r = False
                s,e = i + 1,i + 1
            elif bios[i].startswith('B-') and r:
                segments.append((s, e, bios[i - 1][bios[i - 1].find('-') + 1:]))
                if e - s + 1 > longest: longest = e - s + 1
                s,e = i,i
                r = False
            else:
                e = i
                r = True
        if r: segments.append((s, e, bios[-1][bios[-1].find('-') + 1:]))
        return segments,longest

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
    p_opt.initial_embeddings = None
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
    if options.inputs != '' and options.params != '' and options.model != '':
        print options.model, options.params, options.eval_format
        print 'reading params'
        with open(options.params, 'r') as paramsfp:
            words, bio_tags, bios, ch, opt = pickle.load(paramsfp)
        chunker = Chunker(opt, words, bio_tags, bios, ch, tagger)

        print 'loading model'
        print options.model
        chunker.load(options.model)
        files = options.inputs.strip().split(',')
        for f in files:
            print f
            test = list(Tagger.read_raw(f))
            print 'loaded', len(test), 'sentences!'
            writer = codecs.open(f+'.chunk', 'w')
            for i, sent in enumerate(test, 1):
                output = list()
                bio_tags, pos_tags = chunker.chunk_raw_sent(sent)
                [output.append(' '.join([sent[i], pos_tags[i], bio_tags[i]])) for i in xrange(len(bio_tags))]
                writer.write('\n'.join(output))
                writer.write('\n\n')
                if i%100==0:
                    sys.stdout.write(str(i)+'...')
            sys.stdout.write('done!')
            writer.close()