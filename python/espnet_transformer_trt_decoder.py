
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

def PositionalEncoding(network,input,configure,model_dict):
  embedding = network.add_constant(shape=(configure.nvocab,configure.odim),model_dict["decoder.embed.0.weight"]).get_output(0)
  gather = network.add_gather(embedding,input,0).get_output(0)
  position = position_wise(network,input,configure.maxseql,configure.odim,configure.dtype,model_dict,"decoder.embed.1.pe")

  return position

def DecoderLayer(network,input,encoder,configure,model_dict,prefix):
  tmp = input
  if configure.normalize_before:
    tmp = layer_normalization(network, tmp,
    configure.odim, model_dict, prefix + ".norm1")

  self = self_attention(network, tmp, configure.n_Head,
    configure.odim, model_dict, prefix)

  if configure.concat_after:
    concat = network.add_concatenation([tmp,self]).get_output(0)
    fc = FC(network, concat, configure.odim,
      configure.feed_forward, model_dict, prefix + ".concat_linear1")
    tmp = network.add_elementwise(input,fc,trt.ElementWiseOperation.SUM).get_output(0)
  else:
    tmp = network.add_elementwise(input,self,trt.ElementWiseOperation.SUM).get_output(0)

  if not configure.normalize_before:
    tmp = layer_normalization(network, tmp,
    configure.odim, model_dict, prefix + ".norm1")

  input = tmp
  if configure.normalize_before:
    tmp = layer_normalization(network, tmp,
    configure.odim, model_dict, prefix + ".norm2")

  src = src_attention(network,tmp,encoder,configure.n_Head
    configure.odim, odim.dtype, model_dict, prefix)

  if configure.concat_after:
    concat = network.add_concatenation([tmp,src]).get_output(0)
    fc = FC(network, concat, configure.odim,
      configure.feed_forward, model_dict, prefix + ".concat_linear2")
    tmp = network.add_elementwise(input,fc,trt.ElementWiseOperation.SUM).get_output(0)
  else:
    tmp = network.add_elementwise(input,src,trt.ElementWiseOperation.SUM).get_output(0)
  
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
  if not configure.normalize_before:
    tmp = layer_normalization(network, tmp,
      configure.odim, model_dict, prefix + ".norm3")
  return tmp

def DecoderLayers(network,input,encoder,configure,model_dict):
  for i in range(configure.decoder_layers):
    input = DecoderLayer(network, input, encoder, configure,
	model_dict, "decoder.decoders." + str(i))
  return input

def Espnet_TRT_Transformer_Decoder(configure):
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(TRT_LOGGER)
  network = builder.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  
  model_dict = torch.load("",map_location='cpu')
  input = network.add_input("words",dtype = trt.DataType.INT32,shape=(-1))
  encoder = network.add_input("encoder",dtype = configure.dtype,shape=(-1,configure.odim))
  
  bottom = PositionalEncoding(network,input,configure,model_dict)
  bottom = DecoderLayers(network,bottom,encoder,configure,model_dict)
  finalslice = final_slice(network,bottom)
  shuffle = network.add_shuffle(finalslice)
  shuffle.reshape_dims = (1,configure.odim)
  if configure.normalize_before:
    bottom = layer_normalization(network, shuffle.get_output(0),
      configure.odim, model_dict, "decoder.after_norm")
  softmax = network.add_softmax(bottom).get_output(0)
  topk = network.add_topk(softmax,trt.TopKOperation.MAX,configure.topk,2)
  prob = topk.get_output(0)
  index = topk.get_output(1)
  prob.set_name("prob")
  index.set_name("index")
  
  network.mark_output(prob)
  network.mask_output(index)
  
  config = builder.create_builder_config()
  config.set_flag(trt.BuilderFlag.FP16)
  profile = builder.create_optimization_profile();
  profile.set_shape("words", (1), (64), (192))
  profile.set_shape("encoder",(100,configure.odim),(500,configure.odim),(1600,configure.odim))
  config.add_optimization_profile(profile)

  builder.max_workspace_size = 2**30
  builder.max_batch_size = configure.batchsize
  engine = builder.build_engine(network,config)

  with open("", "wb") as f:
    f.write(engine.serialize())
