
import math
import tensorrt as trt
from util import FC
from util import LayerNormalization
from uitl import SelfAttention
from util import PositionWise

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
  shuffle.first_transpose = (0, 2, 3, 1)
  shuffle.reshape_dims = (-1,configure.odim)
  
  bottom = position_wise(network, shuffle.get_output(0),
    configure.maxseql, configure.odim, configure.dtype,
    model_dict,"encoder.embed.out.1.pe")
  return bottom

def EncoderLayer(network,input,configure,model_dict,prefix):
  tmp = input
  if configure.normalize_before:
    tmp = layer_normalization(network, tmp,
      configure.odim, model_dict, prefix + ".norm1")
  self = self_attention(network, tmp,
    configure.n_Head, configure.odim,
    1, model_dict, prefix)

  if configure.concat_after:
    concat = network.add_concatenation([tmp,self])
    concat.axis = 1
    tmp = FC(network, concat.get_output(0),
      configure.odim, configure.odim,
      model_dict, prefix + ".concat_linear1")
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
  for i in range(configure.encoder_layers):
    input = EncoderLayer(network, input, configure,
	model_dict,"encoder.encoders." + str(i))

  if configure.normalize_before:
    input = layer_normalization(network, input,
      configure.odim, model_dict, "encoder.after_norm")

  return input

def Espnet_TRT_Transformer_Encoder(configure,model_dict):
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(TRT_LOGGER)
  network = builder.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

  input = network.add_input("data",dtype = configure.dtype,shape=(1,1,configurep.odim,-1))
  bottom = Conv2dSubsampling(network,input,model_dict,configure)
  bottom = EncoderLayers(network, input, model_dict, configure)
  
  #CTC Prob
  {
    ctc_fcn = FC(network,bottom,configure.odim,configure.nvocab,model_dict,"ctc.ctc_lo").get_output(0)
    ctc_softmax = network.add_softmax(ctc_fcn)
    ctc_softmax.axes = 2
	ctc_log = network.add_unary(ctc_softmax.get_output(0),trt.UnaryOperation.Log).get_output(0)
	ctc_log.set_name("log_ctc_prob")
	network.mark_output(ctc_log)
  }
  
  bottom.set_name("encoder")
  network.mark_output(bottom)

  config = builder.create_builder_config()
  config.set_flag(trt.BuilderFlag.FP16)
  profile = builder.create_optimization_profile();
  profile.set_shape("data", (1, 1, configure.idim, 500), (1, 1, configure.idim, 2000), (1, 1, configure.idim, 6000)) 
  config.add_optimization_profile(profile)

  builder.max_workspace_size = 2**30
  builder.max_batch_size = 1
  engine = builder.build_engine(network,config)

  with open("", "wb") as f:
    f.write(engine.serialize())
