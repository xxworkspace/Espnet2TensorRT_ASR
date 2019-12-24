
import sys
import torch
import tensorrt as trt

def get_plugin_creator(plugin_name):
  plugin_creators = trt.get_plugin_registry().plugin_creator_list
  for plugin_creator in plugin_creators:
    if plugin_creator.name == plugin_name:
      return plugin_creator
  sys.exit("no plugin find in creators")

def position_wise(network,input,seql,odim,dtype,model_dict,prefix):
  creator = get_pulgin_creator("PositionWise_TRT")
  args = trt.PluginField("args",[seql,odim,(int)dtype], trt.PluginFieldType.INT32)
  pe = trt.PluginField("pe",model_dict[prefix], trt.PluginFieldType.FLOAT32)
  pfc = trt.PluginFieldCollection([args,pe])
  plugin = creator.create_plugin(name="PositionWise_TRT", field_collection=pfc)
  bottom = network.add_plugin_v2(inputs=[input], plugin=plugin).get_output(0)
  return bottom

def FC(network,input,insize,outsize,model_dict,prefix):
  weight = trt.Weights(model_dict[prefix + ".weight"])
  bias = trt.Weights(model_dict[prefix + ".bias"])
  w = network.add_constant(shape=(outsize,insize),weights=weight).get_output(0)
  b = network.add_constant(shape=(1,outsize),weights=bias).get_output(0)
  fc = network.add_matrix_multiply(input,trt.MatrixOperation.NONE,w,trt.MatrixOperation.TRANSPOSE).get_output(0)
  fb = network.add_elementwise(fc,b,trt.ElementWiseOperation.SUM).get_output(0)
  return fb

def layer_normalization(network,input,odim,model_dict,prefix):
  creator = get_pulgin_creator("LayerNormalization_TRT")
  gamma = trt.PluginField("gamma",model_dict[prefix + ".weight"], trt.PluginFieldType.FLOAT32)
  beta = trt.PluginField("beta",model_dict[prefix + ".bias"], trt.PluginFieldType.FLOAT32)
  pfc = trt.PluginFieldCollection([gamma,beta])
  plugin = creator.create_plugin(name="LayerNormalization_TRT", field_collection=pfc)
  bottom = network.add_plugin_v2(inputs=[input], plugin=plugin)
  return bottom

def feed_forward(network,input,odim,feed,model_dict,prefix):
  fc1 = FC(network,input,odim,feed,prefix + ".feed_forward.w_1")
  fb1 = network.add_activation(fc1,trt.ActivationType.RELU).get_output(0)
  fc2 = FC(network,fb1,odim,feed,prefix + ".feed_forward.w_2")
  return fc2

def self_attention(network,input,n_head,odim,model_dict,prefix):
  q = FC(network, input, odim, odim, model_dict, prefix + ".self_attn.linear_q")
  k = FC(network, input, odim, odim, model_dict, prefix + ".self_attn.linear_k");
  v = FC(network, input, odim, odim, model_dict, prefix + ".self_attn.linear_v");
  
  creator = get_pulgin_creator("SelfAttention_TRT")
  args = trt.PluginField("args",[n_head,odim], trt.PluginFieldType.INT32)
  pfc = trt.PluginFieldCollection([args])
  plugin = creator.create_plugin(name="SelfAttention_TRT", field_collection=pfc)
  bottom = network.add_plugin_v2(inputs=[q,k,v], plugin=plugin).get_output(0)
  out = FC(network, bottom, odim, odim, model_dict, prefix + ".self_attn.linear_out")
  return out

def src_attention(network,input,encoder,n_head,odim,dtype,model_dict,prefix):
  q = FC(network, input, odim, odim, model_dict, prefix + ".src_attn.linear_q");

  creator = get_pulgin_creator("SrcAttention_TRT")
  args = trt.PluginField("args",[n_head,odim,(int)dtype], trt.PluginFieldType.INT32)
  kw = trt.PluginField("kweight",model_dict[prefix + ".src_attn.linear_k.weight"],trt.PluginFieldType.FLOAT32)
  kb = trt.PluginField("kbias",model_dict[prefix + ".src_attn.linear_k.bias"],trt.PluginFieldType.FLOAT32)
  vw = trt.PluginField("vweight",model_dict[prefix + ".src_attn.linear_v.weight"],trt.PluginFieldType.FLOAT32)
  vb = trt.PluginField("vbias",model_dict[prefix + ".src_attn.linear_v.bias"],trt.PluginFieldType.FLOAT32)
  pfc = trt.PluginFieldCollection([args,kw,kb,vw,vb])
  plugin = creator.create_plugin(name="SrcAttention_TRT", field_collection=pfc)
  bottom = network.add_plugin_v2(inputs=[q,encoder], plugin=plugin).get_output(0)
  out = FC(network, bottom, odim, odim, model_dict, prefix + ".src_attn.linear_out");
  return out

def final_slice(network,input):
  creator = get_pulgin_creator("FinalSlice_TRT")
  pfc = trt.PluginFieldCollection([])
  plugin = creator.create_plugin(name="FinalSlice_TRT", field_collection=pfc)
  bottom = network.add_plugin_v2(inputs=[input], plugin=plugin)
  return bottom
