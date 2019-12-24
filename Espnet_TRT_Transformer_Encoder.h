#pragma once

#include <map>
#include "util.h"
#include <vector>
#include <assert.h>
#include "logger.h"
#include <NvInfer.h>

using namespace nvinfer1;

ITensor* Conv2dSubsampling(
  INetworkDefinition *network,ITensor* input,
  std::map<std::string, std::string> configure) {
  int idim = std::stoi(configure["idim"]);
  int odim = std::stoi(configure["odim"]);

  auto kernel0 = getWeight("encoder.embed.conv.0.weight",1 * odim * 9 * sizeof(float));
  auto bias0= getWeight("encoder.embed.conv.0.bias", odim * sizeof(float));
  auto conv0 = network->addConvolution(*input,odim,DimsHW(3,3), kernel0, bias0);
  conv0->setStride(DimsHW(2,2));
  auto act0 = network->addActivation(*conv0->getOutput(0),ActivationType::kRELU)->getOutput(0);

  auto kernel1 = getWeight("encoder.embed.conv.2.weight",odim * odim * 9 * sizeof(float));
  auto bias1 = getWeight("encoder.embed.conv.2.bias", odim * sizeof(float));
  auto conv1 = network->addConvolution(*act0, odim, DimsHW(3, 3), kernel1, bias1);
  conv1->setStride(DimsHW(2, 2));
  auto avt1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU)->getOutput(0);

  auto kernel2 = getWeight("encoder.embed.out.0.weight", odim * odim * 20 * sizeof(float));
  auto bias2 = getWeight("encoder.embed.out.0.bias", odim * sizeof(float));
  auto conv2 = network->addConvolution(*avt1,odim,DimsHW(20,1), kernel2, bias2);
  conv2->setStride(DimsHW(20, 1));

  auto shuffle = network->addShuffle(*conv2->getOutput(0));
  shuffle->setFirstTranspose(Permutation{ 1,2,0 });
  shuffle->setReshapeDimensions(DimsHW(-1,odim));

  DataType ctype = configure["trt_dtype"] == "Float" ? DataType::kFLOAT : DataType::kHALF;
  auto bottom = PositionWise(network,shuffle->getOutput(0),std::stoi(configure["maxseql"]),odim, ctype,"encoder.embed.out.1.pe");

  return bottom;
  //float data[] = { sqrtf(odim) };
  //Weights scale{ DataType::kFLOAT, data, 1 };
  //Weights shift{ DataType::kFLOAT ,NULL,0 };
  //Weights power{ DataType::kFLOAT ,NULL,0 };
  //auto scalar = network->addScale(*shuffle->getOutput(0),
  //  ScaleMode::kUNIFORM, shift, scale, power);

  //DataType ctype = configure["trt_dtype"] == "Float" ? DataType::kFLOAT : DataType::kHALF;
  //network->addShape();
  //auto position = network->addInput("position", ctype, DimsHW(-1,odim));

  //auto ele = network->addElementWise(*scalar->getOutput(0),*position,ElementWiseOperation::kSUM);

  //return ele->getOutput(0);
}

ITensor* Encoder_Layer(
  INetworkDefinition *network, 
  ITensor* input,std::string prefix,
  std::map<std::string, std::string> configure) {
  int odim = std::stoi(configure["odim"]);
  int n_head = std::stoi(configure["n_Head"]);
  int feed_forward = std::stoi(configure["feed_forward"]);

  bool concat_after = configure["concat_after"] == "True";
  bool normalize_before = configure["normalize_before"] == "True";
  auto tmp = input;
  if (normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + "norm1.");

  auto self = SelfAttention(network, tmp, n_head, odim, prefix);
  if (concat_after) {
    std::vector<ITensor*> vit(tmp, self);
    auto concat = network->addConcatenation(vit.data(), vit.size());
    concat->setAxis(1);
    tmp = FC(network, concat->getOutput(0), odim, odim, prefix);

    tmp = network->addElementWise(*input, *tmp, ElementWiseOperation::kSUM)->getOutput(0);
  }else
    tmp = network->addElementWise(*input, *tmp, ElementWiseOperation::kSUM)->getOutput(0);

  if(!normalize_before)
    tmp = LayerNormalization(network, tmp, odim,prefix + "norm1.");

  auto input = tmp;
  if (normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + "norm2.");

  tmp = FeedForward(network, tmp, odim, feed_forward, prefix);
  auto ele1 = network->addElementWise(*input, *tmp, ElementWiseOperation::kSUM);

  if(!normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + "norm2.");

  return tmp;
}

ITensor* Encoder_Layers(
  INetworkDefinition *network, ITensor* input,
  std::map<std::string, std::string> configure){

  int odim = std::stoi(configure["odim"]);
  int attention_layers = std::stoi(configure["attention"]);

  for (int i = 0; i < attention_layers; ++i)
    input = Encoder_Layer(network, input, "encoder.encoders." + std::to_string(i) + ".", configure);

  if (configure["normalize_before"] == "True")
    input = LayerNormalization(network, input, odim, "encoder.after_norm.");

  return input;
}

void Espnet_TRT_Transformer_Encoder(
 std::map<std::string,std::string> configure) {
  Logger logger;
  IBuilder* builder = createInferBuilder(logger.getTRTLogger());
  assert(builder == NULL);
  INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  assert(network == NULL);

  int idim = std::stoi(configure["idim"]);
  int odim = std::stoi(configure["odim"]);
  int nvocab = std::stoi(configure["nvocab"]);
  DataType ctype = configure["trt_dtype"] == "Float" ? DataType::kFLOAT : DataType::kHALF;

  //input_layer == "conv2d":
  auto input = network->addInput("data", ctype, DimsCHW(1, idim, -1));
  auto bottom = Conv2dSubsampling(network, input, configure);

  bottom = Encoder_Layers(network, bottom, configure);
  //CTC Prob
  {
    auto ctc_fcn = FC(network,bottom,odim,nvocab,"ctc.ctc_lo.");
    auto cfc_softmax = network->addSoftMax(*ctc_fcn);
    cfc_softmax->setAxes(2);
    auto ctc_log = network->addUnary(*cfc_softmax->getOutput(0),UnaryOperation::kLOG);
    network->getOutput(0)->setName("log_ctc_prob");
    network->markOutput(*network->getOutput(0));
  }

  bottom->setName("encoder");
  network->markOutput(*bottom);

  builder->setMaxBatchSize(std::stoi(configure["batchsize"]));
  IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("words", OptProfileSelector::kMIN, Dims{ 1, 1 });
  profile->setDimensions("words", OptProfileSelector::kOPT, Dims{ 1, 64 });
  profile->setDimensions("words", OptProfileSelector::kMAX, Dims{ 1, 128 });

  IBuilderConfig* config = builder->createBuilderConfig();
  config->addOptimizationProfile(profile);
  config->setMaxWorkspaceSize(1 < 30);

  builder->setMaxBatchSize(16);
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

  IHostMemory* trtModelStream = engine->serialize();
}
