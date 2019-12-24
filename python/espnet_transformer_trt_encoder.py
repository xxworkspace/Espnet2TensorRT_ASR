
import math
import torch
import argparse
import tensorrt as trt
from util import FC
from util import LayerNormalization
from uitl import SelfAttention
from util import PositionWise

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
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

def Conv2dSubsampling(network,input,model_dict,configure):
  conv0 = network.add_convolution(input,
    num_output_maps = configure.odim, kernel_shape=(3,3),
    kernel = model_dict["encoder.embed.conv.0.weight"],
    bias = model_dict["encoder.embed.conv.0.bias"])
  conv0.stride = (2,2)
  act0 = network.add_activation(conv0.get_output(0),trt.ActivationType.RELU).get_output(0)

  conv1 = network.add_convolution(act0,
    num_output_maps = configure.odim, kernel_shape=(3,3),
    kernel = model_dict["encoder.embed.conv.2.weight"],
    bias=model_dict["encoder.embed.conv.2.bias"])
  conv1.stride = (2,2)
  act1 = network.add_activation(conv1.get_output(0),trt.ActivationType.RELU).get_output(0)

  conv2 = network.add_convolution(act1,
    num_output_maps = configure.odim, kernel_shape=(20,1),
    kernel = model_dict["encoder.embed.out.0.weight"],
    bias = model_dict["encoder.embed.out.0.bias"]);
  conv2.stride = (20,1)

  shuffle = network.add_shuffle(conv2.get_output(0))
  shuffle.first_transpose = (1,2,0)
  shuffle.reshape_dims = (-1,configure.odim)
  
  bottom = position_wise(network, shuffle.get_output(0),
    configure.maxseql, configure.odim, configure.dtype,
    model_dict,"encoder.embed.out.1.pe")
  return bottom

def EncoderLayer(network,input,prefix,model_dict,configure):
  tmp = input
  if configure.normalize_before:
    tmp = layer_normalization(network, tmp,
      configure.odim, model_dict, prefix + ".norm1")
  self = self_attention(network, tmp,
    configure.n_Head, configure.odim, prefix)

  if configure.concat_after:
    concat = network.add_concatenation([tmp,self])
    concat.axis = 1
    tmp = FC(network, concat.get_output(0),
      configure.odim, configure.odim,
      model_dict, prefix + "")
	tmp = network.add_elementwise(input,tmp,trt.ElementWiseOperation.SUM).get_output(0)
  else:
    tmp = network.add_elementwise(input,tmp,trt.ElementWiseOperation.SUM).get_output(0)

  if not configure.normalize_before:
    tmp = layer_normalization(network, tmp,
      configure.odim, model_dict, prefix + ".norm1")

  inpute = tmp
  if configure.normalize_before:
    tmp = layer_normalization(network, tmp,
      configure.odim, model_dict, prefix + ".norm2")

  tmp = feed_forward(network, tmp, configure.odim,
    configure.feed_forward, model_dict, prefix)
  tmp = network.add_elementwise(input, tmp, trt.ElementWiseOperation.SUM).get_output(0)
  if not configure.normalize_before:
    tmp = layer_normalization(network, tmp,
      configure.odim, model_dict, prefix + ".norm2")

  return tmp

def EncoderLayers(network,input,model_dict,configure):
  for i in range(configure.layers):
    input = EncoderLayer(network, input,
	"encoder.encoders." + str(i), model_dict, configure)

  if configure.normalize_before:
    input = layer_normalization(network, input,
      configure.odim, model_dict, "encoder.after_norm")

  return input

def Espnet_TRT_Transformer_Encoder(configure):
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(TRT_LOGGER)
  network = builder.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  
  model_dict = torch.load("",map_location='cpu')
  
  input = network.add_input("data",dtype = configure.dtype,shape=(1,configurep.odim,-1))
  bottom = Conv2dSubsampling(network,input,configure)
  bottom = EncoderLayers(network, input, model_dict, configure)
  
  #CTC Prob
  {
    ctc_fcn = FC(network,bottom,"",configure).get_output(0)
    ctc_softmax = network.add_softmax(ctc_fcn)
    ctc_softmax.axes = 2
	ctc_log = network.add_unary(ctc_softmax.get_outptu(0),trt.UnaryOperation.Log).get_output(0)
	network.mark_output(ctc_log)
  }
  
  bottom.set_name("encoder")
  network.mark_output(bottom)
  