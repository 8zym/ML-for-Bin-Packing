import torch
import torch.nn as nn
from Params import args
import math
import torch.nn.functional as F

class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    """

    def __init__(self, hidden_size,
                 dropout, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size  # size of a single head

    def forward(self, query, key, value):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn = torch.matmul(query, key.transpose(-1, -2))

        attn = attn / math.sqrt(self.hidden_size)  # B x M X (M+L)
        attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)  # B x M X (M+L)

        out = torch.matmul(attn, value)  # B x M x H

        return out

class MultiHeadSeqAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, **kargs):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(
            hidden_size=self.head_dim, nb_heads=nb_heads, dropout=args.dropout, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

        # note that the linear layer initialization in current Pytorch is kaiming uniform init

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x


    def forward(self, query, key, value):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size, dropout, **kargs):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d,
            'layer': nn.LayerNorm
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        self.init_parameters()

    def init_parameters(self):
        # xavier_uniform initialization
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):


            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class TransformerSeqLayer(nn.Module):
    def __init__(self, hidden_size,  normalization, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttention(
            hidden_size=hidden_size,  **kargs)
        self.ff = FeedForwardLayer(hidden_size=hidden_size, dropout=args.dropout, inner_hidden_size=args.inner_hidden_size, **kargs)
        self.norm1 = Normalization(hidden_size, normalization)
        self.norm2 = Normalization(hidden_size, normalization)

        # self.enable_mem = enable_mem

    def forward(self, h, h_all):
        # h = B x M x H
        # h_cache = B x L x H
        attn_out = self.attn(h, h_all, h_all)
        h = self.norm1(h + attn_out)  # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out)  # B x M x H
        return out


class EncoderSeq(nn.Module):
    def __init__(self, state_size, hidden_size, nb_heads, encoder_nb_layers,
                  **kargs):
        nn.Module.__init__(self)
        # init embeddings
        self.init_embed = nn.Linear(state_size, hidden_size)

        self.layers = nn.ModuleList()
        self.layers.extend(
            TransformerSeqLayer(
                hidden_size=hidden_size, nb_heads=nb_heads, normalization=args.normalization,
                 **kargs)
            for _ in range(encoder_nb_layers))

    def forward(self, x):
        # x size = B x M
        # block_size = x.size(1)
        h = self.init_embed(x)  # B x M x H
        # h_cache_next = []
        for l, layer in enumerate(self.layers):
            # B x L x H
            h = layer(h, h)  # B x M x H

        return h


class QDecoder(nn.Module):
    def __init__(self, state_size, hidden_size, nb_heads, decoder_nb_layers,
                 **kargs):
        nn.Module.__init__(self)
        # init embeddings
        self.init_embed = nn.Linear(state_size, hidden_size)

        self.layers = nn.ModuleList()
        self.layers.extend(
            TransformerSeqLayer(
                hidden_size=hidden_size,  nb_heads=nb_heads, normalization=args.normalization
                , **kargs)
            for _ in range(decoder_nb_layers))

    def forward(self, x, embedding):
        # x size = B x Q_M
        block_size = x.size(1)
        h = self.init_embed(x)  # B x Q_M x H
        # h_cache_next = []
        for l, layer in enumerate(self.layers):

            h = layer(h, embedding)  # B x Q_M x H

        return h


def get_tgt_entropy(problem_type, block_size, tgt_entropy, p_options):
    s_tgt_entropy = block_size * tgt_entropy / p_options
    if problem_type=='pack2d':
        r_tgt_entropy = 2 * tgt_entropy / p_options
        target_entropy = torch.tensor([s_tgt_entropy, r_tgt_entropy, tgt_entropy])
    elif problem_type=='pack3d':
        r_tgt_entropy = 6 * tgt_entropy / p_options
        target_entropy = torch.tensor([s_tgt_entropy, r_tgt_entropy])
    else:
        raise ValueError('Invalided problem type')

    print('target_entropy: ', target_entropy)
    return target_entropy



class PackDecoder(nn.Module):
    def __init__(self, head_hidden_size, res_size, state_size, hidden_size, decoder_layers, **kargs):
        nn.Module.__init__(self)

        self.att_decoder = QDecoder(state_size, hidden_size, decoder_nb_layers=decoder_layers, nb_heads=args.nb_heads, **kargs)

        self.head = nn.Sequential(
                            nn.Linear(hidden_size, head_hidden_size),
                            nn.ReLU(),
                            nn.Linear(head_hidden_size, res_size)
                            )


    def forward(self, x, embedding):
        h = self.att_decoder(x, embedding)
        out = self.head(h)
        return out

class SelectDecoder(nn.Module):
    def __init__(self, state_size, hidden_size, decoder_layers, **kargs):
        nn.Module.__init__(self)

        self.att_decoder = QDecoder(state_size, hidden_size, decoder_nb_layers=decoder_layers,nb_heads=args.nb_heads, **kargs)
        self.hidden_size=hidden_size

        self.init_embed = nn.Linear(hidden_size, hidden_size)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)

        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh=nn.Tanh()
    def forward(self, x, embedding):
        h = self.att_decoder(x, embedding)
        h = self.init_embed(h)
        query=h
        query=self.proj_query(query)

        key=self.proj_key(embedding)
        attn = torch.matmul(query, key.transpose(-1, -2))
        attn = attn / math.sqrt(self.hidden_size)
        attn=self.tanh(attn)
        attn=10*attn
        return attn

class Cirtic(nn.Module):
    def __init__(self, head_hidden_size, res_size,state_size,hidden_size, c_encoder_layers,  **kargs):
        nn.Module.__init__(self)
        # add this parameter for entropy temp
        # 3D
        self.log_alpha = nn.Parameter(torch.tensor([-2.0,-2.0]))
        
        self.att_decoder = QDecoder(state_size, hidden_size, decoder_nb_layers=c_encoder_layers, nb_heads=args.nb_heads, **kargs)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Linear(head_hidden_size, res_size)
        )


    def forward(self,  x,embedding):

        h= self.att_decoder(x,embedding)
        out = self.head(h)
        return out

def get_ac_parameters(modules):

    critic_params = modules['critic'].parameters()

    actor_params = modules['actor'].parameters()

    return actor_params, critic_params


class HeightmapEncoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size, map_size):
        super(HeightmapEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_size, int(hidden_size/4), stride=2, kernel_size=1)
        self.conv2 = nn.Conv2d(int(hidden_size/4), int(hidden_size/2), stride=2, kernel_size=1)
        self.conv3 = nn.Conv2d(int(hidden_size/2), int(hidden_size), kernel_size=( math.ceil(map_size[0]/4), math.ceil(map_size[1]/4) ) )

    def forward(self, input):

        output = F.leaky_relu(self.conv1(input))
        output = F.leaky_relu(self.conv2(output))
        output = self.conv3(output).squeeze(-1).squeeze(-1).unsqueeze(1)
        return output  # (batch, hidden_size, seq_len)


def build_model(): 
    encoderstate = EncoderSeq(state_size=7,encoder_nb_layers=args.encoder_layers,hidden_size=args.hidden_size,nb_heads=args.nb_heads)

    encoderheightmap=HeightmapEncoder(input_size=2,hidden_size=args.hidden_size,map_size=(args.bin_x,args.bin_y))

    s_decoder = SelectDecoder(state_size=128,hidden_size=args.hidden_size,decoder_layers=args.decoder_layers)

    r_decoder = PackDecoder(head_hidden_size=args.head_hidden,res_size=6,state_size=128,hidden_size=args.hidden_size,decoder_layers=args.decoder_layers)

    critic = Cirtic(head_hidden_size=args.head_hidden,res_size=1,state_size=128,hidden_size=args.hidden_size,c_encoder_layers=args.c_encoder_layers)

    actor_modules = nn.ModuleDict({
        'encoder': encoderstate,
        "encoderheightmap":encoderheightmap,
        's_decoder': s_decoder,
        'r_decoder': r_decoder
    })

    packing_modules = nn.ModuleDict({
        'actor': actor_modules,
        'critic': critic
    })

    return packing_modules

