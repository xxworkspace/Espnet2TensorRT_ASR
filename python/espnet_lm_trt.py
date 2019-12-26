
import torch
import argparse
import tensorrt as trt

parser = argparse.ArgumentParser(description="Espnet LM Parameters!")
parser.add_argument("--model_path",type = str,required=True,help="path to pytorch model!")
parser.add_argument("--embedding",type = int,required=True,help="words embedding size")
parser.add_argument("--nvocab",type = int,required=True,help="the size of vocabulary")
parser.add_argument("--lstm_layers",type = int,required=True,help="number of lstm layers")
parser.add_argument("--hidden_size",type = int, required=True,help="the size of lstm layer")
parser.add_argument("--batchsize",type =int, default = 16,help="the max batchsize of model input")

args = parser.parse_args()
model_dict = torch.load(args.model_path,map_location='cpu')

def embedding(network, input, config, weights):
  embed = network.add_constant(shape(config.nvocab,config.embedding),weights[""])->get_output(0)
  gather = network.add_gather(embed,input,0).get_output(0)
  
  return gather

add_rnn_v2(self: tensorrt.tensorrt.INetworkDefinition, 
input: tensorrt.tensorrt.ITensor,
 layer_count: int,
 hidden_size: int, 
 max_seq_length: int,
 op: tensorrt.tensorrt.RNNOperation) → tensorrt.tensorrt.IRNNv2Layer
 
def lstm(network, input, config, weights):
  rnn = network.add_rnn_v2(input,config.lstm_layers,config.hiden_size,1,trt.RNNOperation.LSTM)
  rnn.input_mode = trt.RNNInputMode.LINEAR
  rnn.direction = trt.RNNDirection.UNIDIRECTION
  
  hidden_cell = network.add_input("hidden_cell",dtype = config.dtype,
  shape=(2*config.lstm_layers,config.hiden_size)).get_output(0)
  hidden = network.add_slice(hidden_cell,start=(0,0),
           shape=(config.lstm_layers,config.hiden_size),stride=(1,1)).get_output(0)
  cell = network.add_slice(hidden_cell,start=(config.lstm_layers,0),
           shape=(config.lstm_layers,config.hiden_size),stride=(1,1)).get_output(0)
  rnn.hidden_state = hidden
  rnn.cell_state = cell
  
  for layerindex in range(config.lstm_layers):
    rnn.set_weights_for_gate(layerindex,trt.RNNGateType.INPUT,True,weights[""])
    rnn.set_weights_for_gate(layerindex,trt.RNNGateType.INPUT,False,weights[""])
    rnn.set_weights_for_gate(layerindex,trt.RNNGateType.FORGET,True,weights[""])
    rnn.set_weights_for_gate(layerindex,trt.RNNGateType.FORGET,False,weights[""])
    rnn.set_weights_for_gate(layerindex,trt.RNNGateType.CELL,True,weights[""])
    rnn.set_weights_for_gate(layerindex,trt.RNNGateType.CELL,False,weights[""])
    rnn.set_weights_for_gate(layerindex,trt.RNNGateType.OUTPUT,True,weights[""])
    rnn.set_weights_for_gate(layerindex,trt.RNNGateType.OUTPUT,False,weights[""])

    rnn.set_bias_for_gate(layerindex,trt.RNNGateType.INPUT,True,weights[""])
    rnn.set_bias_for_gate(layerindex,trt.RNNGateType.INPUT,False,weights[""])
    rnn.set_bias_for_gate(layerindex,trt.RNNGateType.FORGET,True,weights[""])
    rnn.set_bias_for_gate(layerindex,trt.RNNGateType.FORGET,False,weights[""])
    rnn.set_bias_for_gate(layerindex,trt.RNNGateType.CELL,True,weights[""])
    rnn.set_bias_for_gate(layerindex,trt.RNNGateType.CELL,False,weights[""])
    rnn.set_bias_for_gate(layerindex,trt.RNNGateType.OUTPUT,True,weights[""])
    rnn.set_bias_for_gate(layerindex,trt.RNNGateType.OUTPUT,False,weights[""])

  hidden = rnn.get_output(1)
  cell = rnn.get_output(2)
  hidden_cell = network.add_concatenation([hidden,cell]).get_output(0)
  hidden_cell.set_name("hidden_cell_output")
  network.mark_output(hidden_cell)

  return rnn.get_output(0)

def output(network, input, config, weights):
  weight = network.add_constant(shape=(config.nvocab,config.hidden_size),weights[""]).get_output(0)
  bias = network.add_constant(shape=(1, config.nvocab),weights[""]).get_output(0)

  fc = network.add_matrix_multiply(input,trt.MatrixOperation.NONE,weight,trt.MatrixOperation.TRANSPOSE).get_output(0)
  fb = network.add_elementwise(fc,bias,trt.ElementWiseOperation.SUM).get_output(0)

  return fb

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

input = network.add_input("data",dtype = config.dtype,shape=(1))
bottom = embedding(network, input, config, model_dict)
bottom = lstm(network, bottom, config, model_dict)
bottom = output(network,bottom, config, model_dict)
network.add_topk(bottom,,trt.TopKOperation.MAX,config.topk,2)

prob = topk.get_output(0)
index = topk.get_output(1)
prob.set_name("prob")
index.set_name("index")

network.mark_output(prob)
network.mark_output(index)

bconfig = builder.create_builder_config()
bconfig.set_flag(trt.BuilderFlag.FP16)
builder.max_workspace_size = 2**30
builder.max_batch_size = config.batchsize
engine = builder.build_engine(network,bconfig)

with open("", "wb") as f:
  f.write(engine.serialize())
