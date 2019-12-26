#pragma once

#include <map>
#include "util.h"
#include <vector>
#include <cassert>
#include "logger.h"
#include <NvInfer.h>

using namespace nvinfer1;


ITensor* Conv2dSubsampling(
  INetworkDefinition *network,ITensor* input,
  std::map<std::string, std::string> configure) {
  int idim = std::stoi(configure["--idim"]);
  int odim = std::stoi(configure["--odim"]);

  auto kernel0 = getWeight(configure["--path"] + "/encoder.embed.conv.0.weight",1 * odim * 9 * sizeof(float));
  auto bias0= getWeight(configure["--path"] + "/encoder.embed.conv.0.bias", odim * sizeof(float));
  auto conv0 = network->addConvolutionNd(*input,odim,DimsHW(3,3), kernel0, bias0);
  conv0->setStride(DimsHW(2,2));
  auto act0 = network->addActivation(*conv0->getOutput(0),ActivationType::kRELU)->getOutput(0);
  //logTensorInfo(act0,"act0");

  auto kernel1 = getWeight(configure["--path"] + "/encoder.embed.conv.2.weight",odim * odim * 9 * sizeof(float));
  auto bias1 = getWeight(configure["--path"] + "/encoder.embed.conv.2.bias", odim * sizeof(float));
  auto conv1 = network->addConvolutionNd(*act0, odim, DimsHW(3, 3), kernel1, bias1);
  conv1->setStride(DimsHW(2, 2));
  auto avt1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU)->getOutput(0);
  //logTensorInfo(avt1, "avt1");

  auto kernel2 = getWeight(configure["--path"] + "/encoder.embed.out.0.weight", odim * odim * 20 * sizeof(float));
  auto bias2 = getWeight(configure["--path"] + "/encoder.embed.out.0.bias", odim * sizeof(float));
  auto conv2 = network->addConvolutionNd(*avt1,odim,DimsHW(20,1), kernel2, bias2);
  conv2->setStride(DimsHW(20, 1));
  //logTensorInfo(conv2->getOutput(0), "conv2");

  auto shuffle = network->addShuffle(*conv2->getOutput(0));
  shuffle->setFirstTranspose(Permutation{ 0, 2, 3, 1 });
  shuffle->setReshapeDimensions(DimsHW(-1,odim));
  //logTensorInfo(shuffle->getOutput(0), "shuffle");

  DataType ctype = configure["--dtype"] == "float" ? DataType::kFLOAT : DataType::kHALF;
  auto bottom = PositionWise(network,shuffle->getOutput(0),std::stoi(configure["--maxseql"]),
    odim, ctype, configure["--path"] + "/encoder.embed.out.1.pe");

  //logTensorInfo(bottom, "PositionWise");
  return bottom;
}

ITensor* Encoder_Layer(
  INetworkDefinition *network, 
  ITensor* input,std::string prefix,
  std::map<std::string, std::string> configure) {
  int odim = std::stoi(configure["--odim"]);
  int n_head = std::stoi(configure["--n_Head"]);
  int feed_forward = std::stoi(configure["--feed_forward"]);

  bool concat_after = configure["--concat_after"] == "true";
  bool normalize_before = configure["--normalize_before"] == "true";
  auto tmp = input;
  if (normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + ".norm1");

  auto self = SelfAttention(network, tmp, n_head, odim, 0, prefix);
  if (concat_after) {
    std::vector<ITensor*> vit{ tmp, self };
    auto concat = network->addConcatenation(vit.data(), vit.size());
    concat->setAxis(1);
    tmp = FC(network, concat->getOutput(0), odim, odim, prefix + ".concat_linear1");

    tmp = network->addElementWise(*input, *tmp, ElementWiseOperation::kSUM)->getOutput(0);
  }else
    tmp = network->addElementWise(*input, *tmp, ElementWiseOperation::kSUM)->getOutput(0);

  if(!normalize_before)
    tmp = LayerNormalization(network, tmp, odim,prefix + ".norm1");

  input = tmp;
  if (normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + ".norm2");

  tmp = FeedForward(network, tmp, odim, feed_forward, prefix);
  tmp = network->addElementWise(*input, *tmp, ElementWiseOperation::kSUM)->getOutput(0);

  if(!normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + ".norm2");

  return tmp;
}

ITensor* Encoder_Layers(
  INetworkDefinition *network, ITensor* input,
  std::map<std::string, std::string> configure){

  int odim = std::stoi(configure["--odim"]);
  int layers = std::stoi(configure["--encoder_layers"]);

  for (int i = 0; i < layers; ++i)
    input = Encoder_Layer(network, input, configure["--path"] + "/encoder.encoders." + std::to_string(i), configure);

  if (configure["--normalize_before"] == "true")
    input = LayerNormalization(network, input, odim, configure["--path"] + "/encoder.after_norm");

  return input;
}

void Espnet_TRT_Transformer_Encoder(
 std::map<std::string,std::string> configure) {
  Logger logger;
  IBuilder* builder = createInferBuilder(logger.getTRTLogger());
  assert(builder != NULL);
  INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  assert(network != NULL);

  int idim = std::stoi(configure["--idim"]);
  int odim = std::stoi(configure["--odim"]);
  int nvocab = std::stoi(configure["--nvocab"]);
  DataType ctype = configure["--dtype"] == "float" ? DataType::kFLOAT : DataType::kHALF;

  //input_layer == "conv2d":
  auto input = network->addInput("data", ctype, DimsNCHW(1, 1, idim, -1));
  auto bottom = Conv2dSubsampling(network, input, configure);

  bottom = Encoder_Layers(network, bottom, configure);
  //CTC Prob
  {
    auto ctc_fcn = FC(network, bottom, odim, nvocab, configure["--path"] + "/ctc.ctc_lo");
    auto cfc_softmax = network->addSoftMax(*ctc_fcn);
    auto ctc_log = network->addUnary(*cfc_softmax->getOutput(0), UnaryOperation::kLOG);
    ctc_log->setName("log_ctc_prob");
    network->markOutput(*ctc_log->getOutput(0));
  }

  bottom->setName("encoder");
  network->markOutput(*bottom);

  builder->setMaxBatchSize(1);
  IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("data", OptProfileSelector::kMIN, DimsNCHW(1, 1, idim, 500));
  profile->setDimensions("data", OptProfileSelector::kOPT, DimsNCHW(1, 1, idim, 2000));
  profile->setDimensions("data", OptProfileSelector::kMAX, DimsNCHW(1, 1, idim, 6000));

  IBuilderConfig* config = builder->createBuilderConfig();
  config->addOptimizationProfile(profile);
  config->setMaxWorkspaceSize(1 < 30);
  if(configure["--dtype"] == "half")
    config->setFlag(BuilderFlag::kFP16);

  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  IHostMemory* model = engine->serialize();
  std::ofstream ofs(configure["--model_name"] + "_encoder_" + configure["--dtype"] + ".trt", std::ios::binary);
  if (ofs.fail()) {
    std::cerr << "trt model file open fail!" << std::endl;
    exit(0);
  }
  ofs.write((char*)model->data(), model->size());
  ofs.close();

  network->destroy();
  model->destroy();
  engine->destroy();
  builder->destroy();
}
