
import math
import torch
import argparse
import tensorrt as trt

from util import FC
from util import layer_normalization
from uitl import self_attention
from util import src_attention
from util import position_wise
from util import final_slice

parser = argparse.ArgumentParser(description="Espnet Encoder Parameters!")
parser.add_argument("--batchsize",type = int,default=1,help="max batch size!")
parser.add_argument("--idim",type = int,default=83,help="input feature dimension!")
parser.add_argument("--odim",type = int,default=256,help="output feature dimension!")
parser.add_argument("--feed_forward",type = int,default=2048,help="feed forward dimension!")
parser.add_argument("--layers",type = int,default=12,help="attention layers!")
parser.add_argument("--attn_head",type = int,default=4,help="attention layers!")
parser.add_argument("--nvocab",type = int,default=7244,help="the number of vocabularies!")
parser.add_argument("--dtype",type = int,default=0,help="compute data type!")
parser.add_argument("--normalize_before",type = bool,default=True,help="layer normalization before!")
parser.add_argument("--concat_after",type = bool,default=True,help="layer concat!")

args = parser.parse_args()


def PositionalEncoding(network,input,model_dict,configure):
  embedding = network.add_constant(shape=(configure.nvocab,configure.odim),model_dict["decoder.embed.0.weight"]).get_output(0)
  gather = network.add_gather(embedding,input,0).get_output(0)
  position = position_wise(network,input,configure.maxseql,configure.odim,configure.dtype,model_dict,"decoder.embed.1.pe")

  return position

def DecoderLayer(network,input,encoder,model_dict,prefix,configure):
  tmp = input
  if configure.normalize_before:
    tmp = layer_normalization(network, tmp,
    configure.odim, model_dict, prefix + "norm1.")
  self = self_attention(network, tmp, configure.n_Head,
    configure.odim, model_dict, prefix)
  if configure.concat_after:
    concat = network.add_concatenation([tmp,self]).get_output(0)
    fc = FC(network, concat, configure.odim,
      configure.feed_forward, model_dict, prefix + ".concat_linear1")
    tmp = network.add_elementwise(fc,input,trt.ElementWiseOperation.SUM).get_output(0)
  else:
    tmp = network.add_elementwise(self,input,trt.ElementWiseOperation.SUM).get_output(0)

  if not configure.normalize_before:
    tmp = layer_normalization(network, tmp,
    configure.odim, model_dict, prefix + "norm1")

  input = tmp
  if configure.normalize_before:
    tmp = layer_normalization(network, tmp,
    configure.odim, model_dict, prefix + "norm2")
  src = src_attention(network,tmp,encoder,configure.n_Head
    configure.odim, odim.dtype, model_dict, prefix)

  if configure.concat_after:
    concat = network.add_concatenation([tmp,src]).get_output(0)
    fc = FC(network, concat, configure.odim,
      configure.feed_forward, model_dict, prefix + ".concat_linear2")
    tmp = network.add_elementwise(fc,input,trt.ElementWiseOperation.SUM).get_output(0)
  else:
    tmp = network.add_elementwise(src,input,trt.ElementWiseOperation.SUM).get_output(0)
  
  if not configure.normalize_before:
    tmp = layer_normalization(network, tmp,
      configure.odim, model_dict, prefix + ".norm2")

  input = tmp
  if configure.normalize_before:
    tmp = layer_normalization(network, tmp,
      configure.odim, model_dict, prefix + ".norm3")

  tmp = feed_forward(network, tmp, configure.odim,
    configure.feed_forward, model_dict, prefix)
  tmp = network.add_elementwise(tmp,input,trt.ElementWiseOperation.SUM).get_output(0)
  if configure.normalize_before:
    tmp = layer_normalization(network, tmp,
      configure.odim, model_dict, prefix + ".norm3")
  return tmp

def DecoderLayers(network,input,encoder,model_dict,configure):
  for i in range(configure.layers):
    input = DecoderLayer(network,input,encoder,model_dict,
	"decoder.decoders." + str(i),configure)
  return input

def Espnet_TRT_Transformer_Decoder(configure):
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(TRT_LOGGER)
  network = builder.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  
  model_dict = torch.load("",map_location='cpu')
  input = network.add_input("words",dtype = trt.DataType.INT32,shape=(-1))
  encoder = network.add_input("encoder",dtype = configure.dtype,shape=(-1,configure.odim))
  
  bottom = PositionalEncoding(network,input,model_dict,configure)
  bottom = DecoderLayers(network,bottom,encoder,model_dict,configure)

  finalslice = final_slice(network,bottom)
  if configure.normalize_before:
    bottom = layer_normalization(network, finalslice,
      configure.odim, model_dict, "decoder.after_norm")
  softmax = network.add_softmax(bottom).get_output(0)
  topk = network.add_topk(softmax,trt.TopKOperation.MAX,configure.topk,0)
  prob = topl.get_output(0)
  index = topk.get_output(1)
  prob.set_name("prob")
  index.set_name("index")
  
  network.mark_output(prob)
  network.mask_output(index)
  