#pragma once


#include <string>
#include <cassert>
#include <fstream>
#include <iostream>
#include <NvInfer.h>

using namespace nvinfer1;

char* fromFile(std::string file,size_t size) {
  std::ifstream ifs(file,std::ios::binary);
  if (ifs.fail()) {
    std::cout << file <<" open fail!"<<std::endl;
    exit(0);
  }
  ifs.seekg(0,std::ios::end);
  size_t len = ifs.tellg();
  ifs.seekg(0,std::ios::beg);
  assert(len == size);
  char* tmp_ptr = (char*)malloc(len);
  ifs.read(tmp_ptr,len);
  ifs.close();
  return tmp_ptr;
}

Weights getWeight(std::string file,size_t size) {
  char*tmp = fromFile(file, size);
  Weights weight{ DataType::kFLOAT,tmp,(int64_t)size / sizeof(float) };
  return weight;
}

ITensor* PositionWise(
  INetworkDefinition *network,
  ITensor* input,
  const int seql,
  const int odim,
  const DataType dtype,
  std::string prefix) {

  int data[] = { seql ,odim ,(int)dtype };
  auto pe = getWeight(prefix, seql*odim * sizeof(float));

  std::vector<PluginField> vpf{
    PluginField("args",data,PluginFieldType::kINT32,(int32_t)3),
    PluginField("pe",pe.values,PluginFieldType::kFLOAT32,(int32_t)pe.count)
  };

  PluginFieldCollection pfc{ vpf.size(),vpf.data() };
  auto creator = getPluginRegistry()->getPluginCreator(
    "PositionWise_TRT", "001", "");
  IPluginV2 *plugin = creator->createPlugin("", &pfc);
  auto bottom = network->addPluginV2(&input, 1, *plugin)->getOutput(0);
  return bottom;
}

ITensor* FC(
  INetworkDefinition *network,
  ITensor* input,
  const int insize,
  const int outsize,
  std::string prefix) {

  Weights weight = getWeight(prefix + "weight", insize*outsize * sizeof(float));
  Weights bias = getWeight(prefix + "bais", outsize * sizeof(float));

  auto w = network->addConstant(DimsHW(outsize, insize), weight)->getOutput(0);
  auto b = network->addConstant(DimsHW(1,outsize),bias)->getOutput(0);

  auto fc = network->addMatrixMultiply(input, MatrixOperation::kNONE, *w, MatrixOperation::kTRANSPOSE)->getOutput(0);
  auto fb = network->addElementWise(*fc,*b,ElementWiseOperation::kSUM)->getOutput(0);

  return fb;
}

ITensor* LayerNormalization(
  INetworkDefinition *network,
  ITensor* input,
  const int odim,
  std::string prefix) {

  auto gamma = getWeight(prefix + "weight", odim * sizeof(float));
  auto beta = getWeight(prefix + "bias", odim * sizeof(float));

  std::vector<PluginField> vpf{
    PluginField("gamma",gamma.values,PluginFieldType::kFLOAT32,(int32_t)gamma.count),
    PluginField("beta",beta.values,PluginFieldType::kFLOAT32,(int32_t)beta.count)
  };

  PluginFieldCollection pfc{vpf.size(),vpf.data()};
  auto creator = getPluginRegistry()->getPluginCreator(
    "LayerNormalization_TRT", "001", "");
  IPluginV2 *plugin = creator->createPlugin("", &pfc);
  auto bottom = network->addPluginV2(&input, 1, *plugin)->getOutput(0);
  return bottom;
}

ITensor* FeedForward(
  INetworkDefinition *network,
  ITensor* input,
  const int odim,
  const int feed,
  std::string prefix) {

  auto fc1 = FC(network, input, odim, feed, prefix + "feed_forward.w_1.");
  auto fb1 = network->addActivation(*fc1, ActivationType::kRELU)->getOutput(0);
  auto fc2 = FC(network, fb1, feed, odim, prefix + "feed_forward.w_2.");

  return fc2;
}

ITensor* SelfAttention(
  INetworkDefinition *network,
  ITensor* input,
  const int n_Head,
  const int odim,
  std::string prefix) {

  auto q = FC(network, input, odim, odim, prefix + "self_attn.linear_q.");
  auto k = FC(network, input, odim, odim, prefix + "self_attn.linear_k.");
  auto v = FC(network, input, odim, odim, prefix + "self_attn.linear_v.");

  int data[] = { n_Head, odim };
  std::vector<PluginField> vpf{
    PluginField("args",data,PluginFieldType::kFLOAT32,(int32_t)2)
  };

  PluginFieldCollection pfc{ vpf.size(),vpf.data() };
  auto creator = getPluginRegistry()->getPluginCreator(
    "SelfAttention_TRT", "001", "");
  IPluginV2 *plugin = creator->createPlugin("", &pfc);
  std::vector<ITensor*> vit{ q,k,v };
  auto bottom = network->addPluginV2(vit.data(), vit.size(), *plugin)->getOutput(0);

  auto out = FC(network, input, odim, odim, prefix + "self_attn.linear_out.");
  return out;
}

ITensor* SrcAttention(
  INetworkDefinition* network,
  ITensor* input, ITensor* encoder,
  const int n_Head,
  const int odim,
  const DataType dtype,
  std::string prefix) {
  auto q = FC(network, input, odim, odim, prefix + "src_attn.linear_q.");

  int data[] = { n_Head, odim ,(int)dtype };

  auto kWeight = getWeight(prefix + "src_attn.linear_k.weight",odim*odim*sizeof(float));
  auto kBias = getWeight(prefix + "src_attn.linear_k.bais", odim * sizeof(float));

  auto vWeight = getWeight(prefix + "src_attn.linear_v.weight", odim*odim * sizeof(float));
  auto vBias = getWeight(prefix + "src_attn.linear_v.bais", odim * sizeof(float));

  std::vector<PluginField> vpf{
    PluginField("args",data,PluginFieldType::kFLOAT32,(int32_t)2),
    PluginField("kweight",kWeight.values,PluginFieldType::kFLOAT32,(int32_t)kWeight.count),
    PluginField("kbias",kBias.values,PluginFieldType::kFLOAT32,(int32_t)kBias.count),
    PluginField("vweight",vWeight.values,PluginFieldType::kFLOAT32,(int32_t)vWeight.count),
    PluginField("vbias",vBias.values,PluginFieldType::kFLOAT32,(int32_t)vBias.count),
  };

  PluginFieldCollection pfc{ vpf.size(),vpf.data() };
  auto creator = getPluginRegistry()->getPluginCreator(
    "SrcAttention_TRT", "001", "");
  IPluginV2 *plugin = creator->createPlugin("", &pfc);
  std::vector<ITensor*> vit{ q,encoder };
  auto bottom = network->addPluginV2(vit.data(), vit.size(), *plugin)->getOutput(0);

  auto out = FC(network, bottom, odim, odim, prefix + "src_attn.linear_out.");
  return out;
}

ITensor* FinalSlice(
  INetworkDefinition* network,
  ITensor* input,
  std::string prefix) {
  PluginFieldCollection pfc;
  auto creator = getPluginRegistry()->getPluginCreator(
    "FinalSlice_TRT", "001", "");
  IPluginV2 *plugin = creator->createPlugin("", &pfc);
  auto bottom = network->addPluginV2(&input, 1, *plugin)->getOutput(0);
  return bottom;
}