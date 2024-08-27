import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var
from utils import *
from data import *
from lf_evaluator import *
import numpy as np
from typing import List
import time

den_max = 0.0
pad = 0
unk = 1
sos = 1
eos = 2


def add_models_args(parser):
    """
    Command-line arguments to the system related to your model.  Feel free to extend here.  
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=40, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes

    parser.add_argument('--input_dim', type=int, default=125, help='input dimensions')
    parser.add_argument('--output_dim', type=int, default=125, help='output dimensions')
    parser.add_argument('--hidden_size', type=int, default=175, help='hidden state dimensions')
    parser.add_argument('--no_bidirectional', dest='bidirectional', default=True, action='store_false', help='bidirectional flag')
    parser.add_argument('--attn', dest='attn', default=True, action="store_true", help="Attention flag")
    parser.add_argument('--copy', dest='copy', default=False, action="store_true", help="Test if decoder can copy")


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap with Jaccard similarity
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # Note that this is a list of a single Derivation
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


###################################################################################################################
# You do not have to use any of the classes in this file, but they're meant to give you a starting implementation.
# for your network.
###################################################################################################################

class Seq2SeqSemanticParser(object):
    def __init__(self, decoder, encoder, emb_input, emb_output, output_indexer, args, out_max=65):
        self.decoder = decoder
        self.encoder = encoder
        self.emb_input = emb_input
        self.emb_output = emb_output
        self.out_max = out_max
        self.output_indexer = output_indexer

    def decode(self, test_data):
        self.decoder.eval()
        self.encoder.eval()
        self.emb_input.eval()
        self.emb_output.eval()

        derivs = []
        test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
        for i in range(len(test_data)):
            pred, pred_vals = [], []

            input_seq = torch.as_tensor(test_data[i].x_indexed).unsqueeze(0)
            input_len = torch.as_tensor(len(test_data[i].x_indexed)).view(1)

            (output_enc, mask_enc, reshape_enc) = input_encoder(
                input_seq, input_len, self.emb_input, self.encoder)

            input_dec = torch.as_tensor(sos).unsqueeze(0).unsqueeze(0)
            hidden_dec = reshape_enc

            for j in range(self.out_max):
                predict, pred_val, hidden_dec = self.decode_predict(input_dec, hidden_dec)
                input_dec = torch.as_tensor(predict).unsqueeze(0).unsqueeze(0)
                if predict != eos:
                    pred.append(self.output_indexer.get_object(predict))
                    pred_vals.append(pred_val)
                else:
                    break
            derivs.append([Derivation(test_data[i], pred_vals, pred)])

        return derivs

    def decode_predict(self, in_dec, hid_dec):
        out_dec, hid_dec = decode_output(in_dec, hid_dec, self.decoder, self.emb_output)
        pred_val, idx_pred = out_dec.topk(1)

        return int(idx_pred), pred_val, hid_dec


class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """
    def __init__(self, input_dim, full_dict_size, embedding_dropout_rate):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


class RNNEncoder(nn.Module):
    """
    One-layer RNN encoder for batched inputs -- handles multiple sentences at once. To use in non-batched mode, call it
    with a leading dimension of 1 (i.e., use batch size 1)
    """
    def __init__(self, in_size, hidden_size, drop, bidirect):
        """
        :param input_emb_dim: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=1, batch_first=True,
                           dropout=drop, bidirectional=self.bidirect)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens):
        """
        Runs the forward pass of the LSTM
        :param embedded_words: [batch size x sent len x input dim] tensor
        :param input_lens: [batch size]-length vector containing the length of each input sentence
        :return: output (each word's representation), context_mask (a mask of 0s and 1s
        reflecting where the model's output should be considered), and h_t, a *tuple* containing
        the final states h and c from the encoder for each sentence.
        Note that output is only needed for attention, and context_mask is only used for batched attention.
        """
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = input_lens.data[0].item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)

        return output, context_mask, h_t

###################################################################################################################
# End optional classes
###################################################################################################################


class RNNDecoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, drop=0.2, batch_first=True):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.LSTM = nn.LSTM(in_size, hidden_size, num_layers=1, batch_first=batch_first, )
        self.W = nn.Linear(hidden_size, out_size)
        self.log_s_m = nn.LogSoftmax(dim=1)

    def forward(self, embed, hidden_in):
        input_val = self.drop(embed)
        out, hidden_out = self.LSTM(input_val, hidden_in)
        (h_n, c_n) = hidden_out

        return self.log_s_m(self.W(h_n[0])), hidden_out


class AttentionDecoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, args, drop=0.2):
        super().__init__()
        self.args = args
        self.LSTM = nn.LSTM(in_size, hidden_size, )
        self.output = nn.Linear(hidden_size, out_size)
        self.drop = nn.Dropout(drop)
        self.encode_reduce = nn.Linear(2 * hidden_size, hidden_size)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.log_s = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()
        self.s_max = nn.Softmax(dim=1)
        self.s_0 = nn.Softmax(dim=0)
        self.W_c = nn.Linear(hidden_size * 2, hidden_size)
        self.W_s = nn.Linear(hidden_size, out_size)
        self.W_out = nn.Linear(hidden_size * 2, out_size)

    def forward(self, emb, hidden, encode_out):
        if self.args.bidirectional:
            encode_out = self.encode_reduce(encode_out)

        drop_e = self.drop(emb)
        out_i, (h_o, c_o) = self.LSTM(drop_e, hidden)
        encode_out = encode_out.squeeze(1)
        h_x = h_o.squeeze(1)
        e_xy = torch.matmul(h_x, self.attn(encode_out).t())
        attn_w = self.s_max(e_xy)
        c_xy = torch.mm(attn_w, encode_out)

        h_c = torch.cat([h_x, c_xy], dim=1)
        o_attn = self.log_s(self.W_out(h_c))

        return o_attn, (h_o, c_o)


class AttentionParser(object):
    def __init__(self, decoder, encoder, emb_input, emb_output, output_indexer, args, out_max=65):
        self.decoder = decoder
        self.encoder = encoder
        self.emb_input = emb_input
        self.emb_output = emb_output
        self.out_max = out_max
        self.output_indexer = output_indexer

    def decode(self, test_data):
        self.decoder.eval()
        self.encoder.eval()
        self.emb_input.eval()
        self.emb_output.eval()

        derivs = []
        test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
        for i in range(len(test_data)):
            pred, pred_vals = [], []

            input_seq = torch.as_tensor(test_data[i].x_indexed).unsqueeze(0)
            input_len = torch.as_tensor(len(test_data[i].x_indexed)).view(1)

            (output_enc, mask_enc, reshape_enc) = input_encoder(
                input_seq, input_len, self.emb_input, self.encoder)

            input_dec = torch.as_tensor(sos).unsqueeze(0).unsqueeze(0)
            hidden_dec = reshape_enc

            for j in range(self.out_max):
                predict, pred_val, hidden_dec = self.attn_predict(input_dec, hidden_dec , output_enc)
                input_dec = torch.as_tensor(predict).unsqueeze(0).unsqueeze(0)
                if predict != eos:
                    pred.append(self.output_indexer.get_object(predict))
                    pred_vals.append(pred_val)
                else:
                    break
            derivs.append([Derivation(test_data[i], pred_vals, pred)])

        return derivs

    def attn_predict(self, in_dec, hid_dec, out_enc):
        out_dec, hid_dec = output_attention(in_dec, hid_dec, out_enc, self.decoder, self.emb_output)
        pred_val, i_pred = out_dec.topk(1)

        return int(i_pred), pred_val, hid_dec


def input_encoder(tens_x, tens_inp, emb_inp, enc):
    input_emb = emb_inp.forward(tens_x)
    (enc_out, enc_cont, enc_fin) = enc.forward(input_emb, tens_inp)
    enc_reshaped = (enc_fin[0].unsqueeze(0), enc_fin[1].unsqueeze(0))

    return enc_out, enc_cont, enc_reshaped


def forward_attention(data, models, idx, crit, args):

    global correct
    global count_sen

    loss_val = 0.0
    (emb_in, emb_out, enc, dec) = models
    seq_inp = torch.as_tensor(data[idx].x_indexed).unsqueeze(0)
    len_inp = torch.as_tensor(len(data[idx].x_indexed)).view(1)

    if args.copy:
        seq_out = torch.as_tensor(data[idx].x_indexed).view(-1)
        corr, pred = [], []
    else:
        seq_out = torch.as_tensor(data[idx].y_indexed).view(-1)

    (enc_out, enc_cont, enc_reshaped) = input_encoder(seq_inp, len_inp, emb_in, enc)

    dec_inp = torch.as_tensor(sos).unsqueeze(0).unsqueeze(0)
    dec_hid = enc_reshaped

    for idx_o in range(len(seq_out)):
        dec_out, dec_hid = output_attention(dec_inp, dec_hid, enc_out, dec, emb_out)
        pred_val, pred_i = dec_out.topk(1)

        loss_val += crit(dec_out, seq_out[idx_o].unsqueeze(0))

        dec_inp = seq_out[idx_o].unsqueeze(0).unsqueeze(0)
        if args.copy:
            corr.append(int(seq_out[idx_o]))
            pred.append(int(pred_i))

        if int(pred_i) == eos and not args.copy:
            break

    if args.copy:
        count_sen += 1
        if corr == pred:
            correct += 1

    return loss_val


def output_attention(inp_attn, hid_attn, enc_out, attn_mod, emb_out):
    embed = emb_out.forward(inp_attn)

    attn_out, hid_attn = attn_mod.forward(embed, hid_attn, enc_out)
    return attn_out, hid_attn


def decode_forward(data, models, idx, crit, args):

    global correct
    global count_sen

    loss_val = 0.0
    (emb_in, emb_out, enc, dec) = models
    seq_inp = torch.as_tensor(data[idx].x_indexed).unsqueeze(0)
    len_inp = torch.as_tensor(len(data[idx].x_indexed)).view(1)

    if args.copy:
        seq_out = torch.as_tensor(data[idx].x_indexed).view(-1)
        corr, pred = [], []
    else:
        seq_out = torch.as_tensor(data[idx].y_indexed).view(-1)

    (enc_out, enc_cont, enc_reshaped) = input_encoder(seq_inp, len_inp, emb_in, enc)

    dec_inp = torch.as_tensor(sos).unsqueeze(0).unsqueeze(0)
    dec_hid = enc_reshaped

    for idx_o in range(len(seq_out)):
        dec_out, dec_hid = decode_output(dec_inp, dec_hid, dec, emb_out)
        pred_val, pred_i = dec_out.topk(1)

        loss_val += crit(dec_out, seq_out[idx_o].unsqueeze(0))

        dec_inp = seq_out[idx_o].unsqueeze(0).unsqueeze(0)
        if args.copy:
            corr.append(int(seq_out[idx_o]))
            pred.append(int(pred_i))

        if int(pred_i) == eos and not args.copy:
            break

    if args.copy:
        count_sen += 1
        if corr == pred:
            correct += 1

    return loss_val


def decode_output(inp_dec, hid_dec, dec, emb_out):
    embed = emb_out(inp_dec)
    dec_out, hid_out = dec(embed, hid_dec)

    return dec_out, hid_out


def evaluate_2(test_data, decoder, args, example_freq=50, print_output=True, outfile=None):
    """
    Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
    every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
    executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
    example with a valid denotation (if you've provided more than one).
    :param test_data:
    :param decoder:
    :param example_freq: How often to print output
    :param print_output:
    :param outfile:
    :return:
    """
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    use_java = args.perform_java_eval
    if use_java:
        selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations, quiet=True)
    else:
        selected_derivs = [derivs[0] for derivs in pred_derivations]
        denotation_correct = [False for derivs in pred_derivations]
    exact_count = 0
    corr_tok_count = 0
    denot_match_count = 0
    tok_count = 0
    for m, n in enumerate(test_data):
        pred = ' '.join(selected_derivs[m].y_toks)
        if pred == ' '.join(n.y_tok):
            exact_count += 1
        corr_tok_count += sum(a == b for a, b in zip(selected_derivs[m].y_toks, n.y_tok))
        tok_count += len(n.y_tok)
        if denotation_correct[m]:
            denot_match_count += 1
    denot_rat = "%i / %i = %.3f" % (denot_match_count, len(test_data), float(denot_match_count)/len(test_data))
    return denot_rat


def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])


def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


def train_model_encdec(train_data, dev_data, input_indexer, output_indexer, args):
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param dev_data: Development set in case you wish to evaluate during training
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """

    global den_max
    rnn_dropout = 0.2
    emb_dropout = 0.2
    dec_dropout = 0.2

    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    dev_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    mod_emb_inp = EmbeddingLayer(args.input_dim, len(input_indexer), emb_dropout)
    mod_emb_out = EmbeddingLayer(args.output_dim, len(output_indexer), emb_dropout)
    mod_rnn_enc = RNNEncoder(args.input_dim, args.hidden_size, rnn_dropout, args.bidirectional)

    if args.attn:
        model_dec = AttentionDecoder(args.output_dim, args.hidden_size, len(output_indexer), args, drop=dec_dropout)
    else:
        model_dec = RNNDecoder(args.output_dim, args.hidden_size, len(output_indexer), drop=dec_dropout)

    mod_all = (mod_emb_inp, mod_emb_out, mod_rnn_enc, model_dec)

    optim_emb_inp = torch.optim.Adam(mod_emb_inp.parameters(), args.lr)
    optim_emb_out = torch.optim.Adam(mod_emb_out.parameters(), args.lr)
    optim_encoder = torch.optim.Adam(mod_rnn_enc.parameters(), args.lr)
    optim_decoder = torch.optim.Adam(model_dec.parameters(), args.lr)

    crit = torch.nn.NLLLoss()

    start_time = time.time()

    for ep in range(1, args.epochs + 1):

        elapsed_time = time.time() - start_time

        if elapsed_time > 2500:
            print("reached max time, returning from function")
            try:
                return pars_best
            except:
                return pars

        global count_sen
        global correct

        count_sen = 0.0
        correct = 0.0

        mod_emb_out.train()
        mod_emb_inp.train()
        mod_rnn_enc.train()
        model_dec.train()

        tot_loss = 0.0

        for idx_p in range(len(train_data)):

            optim_emb_inp.zero_grad()
            optim_emb_out.zero_grad()
            optim_encoder.zero_grad()
            optim_decoder.zero_grad()

            if args.attn:
                loss_val = forward_attention(train_data, mod_all, idx_p, crit, args)
            else:
                loss_val = decode_forward(train_data, mod_all, idx_p, crit, args)

            tot_loss += loss_val

            loss_val.backward()

            optim_emb_inp.step()
            optim_emb_out.step()
            optim_encoder.step()
            optim_decoder.step()

        print("Epoch, loss, time", ep, tot_loss, elapsed_time)

        if args.attn:
            pars = AttentionParser(model_dec, mod_rnn_enc, mod_emb_inp, mod_emb_out, output_indexer, args)
        else:
            pars = Seq2SeqSemanticParser(model_dec, mod_rnn_enc, mod_emb_inp, mod_emb_out, output_indexer, args)

        if args.copy:
            print("{}% right for copy".format(100 * float(correct / count_sen)))
        else:
            denot = evaluate_2(dev_data, pars, args, print_output=True)
            denot = float(denot.split(" ")[-1])
            if denot > den_max:
                pars_best = pars
                den_max = denot

    if args.copy:
        exit()

    try:
        return pars_best
    except:
        return pars