
import torch
import argparse
from espnet_transformer_trt_encoder import Espnet_TRT_Transformer_Encoder
from espnet_transformer_trt_decoder import Espnet_TRT_Transformer_Decoder

parser = argparse.ArgumentParser(description="Espnet Encoder Parameters!")
parser.add_argument("--model_path",type = str,required=True,help="path to pytorch model!")
parser.add_argument("--idim",type = int,default=83,help="input feature dimension!")
parser.add_argument("--n_Head",type = int,default=4,help="the number of head in attention!")
parser.add_argument("--odim",type = int,default=256,help="the feature dimension!")
parser.add_argument("--feed_forward",type = int,default=2048,help="feed forward dimension!")
parser.add_argument("--nvocab",type = int,required=True,help="the size of vocabulary!")
parser.add_argument("--dtype",type = int,default=0,help="the data type be used for computation!")
parser.add_argument("--concat_after",type = bool,default=False,help="concat is used!")
parser.add_argument("--normalize_before",type = bool,default=True,help="layer normalization before!")
parser.add_argument("--encoder_layers",type = int,default=12,help="the number of attention in encoder!")
parser.add_argument("--decoder_layers",type = int,default=6,help="the number of attention in decoder!")
parser.add_argument("--batchsize",type = int,default=1,help="the max batchsize of decoder!")
parser.add_argument("--topk",type = int,default=16,help="the topk in each decoder step!")
parser.add_argument("--maxseql",type = int,default=5000,help="the max sequence length of encoder!")
parser.add_argument("--model_name",type = int,default=5000,help="the output trt model name!")

args = parser.parse_args()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
model_dict = torch.load(args.model_path,map_location='cpu')

Espnet_TRT_Transformer_Encoder(args,model_dict)
Espnet_TRT_Transformer_Decoder(args,model_dict)
