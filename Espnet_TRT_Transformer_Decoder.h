#pragma once

#include <map>
#include "util.h"
#include <vector>
#include <assert.h>
#include "logger.h"
#include <NvInfer.h>

using namespace nvinfer1;

ITensor* PositionalEncoding(
  INetworkDefinition* network,ITensor* input,
  std::map<std::string, std::string> configure) {
  int nvocab = std::stoi(configure["nvocab"]);
  int odim = std::stoi(configure["odim"]);

  Weights embed = getWeight("decoder.embed.0.weight", nvocab*odim * sizeof(float));
  auto embedding = network->addConstant(DimsHW(nvocab, odim), embed)->getOutput(0);
  auto gather = network->addGather(*embedding, *input, 0)->getOutput(0);
  
  DataType ctype = configure["trt_dtype"] == "Float" ? DataType::kFLOAT : DataType::kHALF;
  auto bottom = PositionWise(network, gather, std::stoi(configure["maxseql"]), odim, ctype, "decoder.embed.1.pe");

  return bottom;
}

ITensor* Decoder_Layer(
  INetworkDefinition *network,
  ITensor* input,
  ITensor* encoder,
  std::string prefix,
  std::map<std::string, std::string> configure) {
  DataType ctype = configure["trt_dtype"] == "Float" ? DataType::kFLOAT : DataType::kHALF;
  bool normalize_before = configure["normalize_before"] == "True";
  bool concat_after = configure["concat_after"] == "True";
  int feed_forward = std::stoi(configure["feed_forward"]);
  int n_head = std::stoi(configure["attn_head"]);
  int odim = std::stoi(configure["odim"]);
  std::string prefix = "";

  auto tmp = input;
  if (normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + "norm1.");

  auto self = SelfAttention(network, tmp, n_head, odim, prefix);
  if (concat_after) {
    std::vector<ITensor*> vit{ tmp, self };
    auto concat = network->addConcatenation(vit.data(), vit.size())->getOutput(0);
    auto fc = FC(network, concat, odim * 2, odim, prefix + "concat_linear1.");
    tmp = network->addElementWise(*input,*fc,ElementWiseOperation::kSUM)->getOutput(0);
  }
  else
    tmp = network->addElementWise(*input,*self, ElementWiseOperation::kSUM)->getOutput(0);

  if (!normalize_before)
    tmp = LayerNormalization(network, tmp, odim,prefix + "norm1.");

  input = tmp;
  if (normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + "norm2.");

  auto src = SrcAttention(network, tmp, encoder, n_head, odim, ctype, prefix);

  if (concat_after) {
    std::vector<ITensor*> vit{ tmp, src };
    auto concat = network->addConcatenation(vit.data(), vit.size())->getOutput(0);
    auto fc = FC(network, concat, odim * 2, odim, prefix + "concat_linear2.");
    tmp = network->addElementWise(*input, *fc, ElementWiseOperation::kSUM)->getOutput(0);
  }else
    tmp = network->addElementWise(*input, *src, ElementWiseOperation::kSUM)->getOutput(0);

  if (!normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + "norm2.");

  input = tmp;
  if (normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + "norm3.");

  tmp = FeedForward(network, tmp, odim, feed_forward, prefix);
  tmp = network->addElementWise(*input, *tmp, ElementWiseOperation::kSUM)->getOutput(0);
  if (!normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + "norm3.");

  return tmp;
}

ITensor* Decoder_Layers(
  INetworkDefinition *network,
  ITensor* input,ITensor* encoder,
  std::map<std::string, std::string> configure) {

  int layers = std::stoi(configure["decoder_layers"]);
  for (int i = 0; i < layers; ++i)
    input = Decoder_Layer(network,input, encoder,"decoder.decoders." + std::to_string(i) + ".",configure);
}

void Espnet_TRT_Transformer_Decoder(
  std::map<std::string, std::string> configure) {

  Logger logger;
  IBuilder* builder = createInferBuilder(logger.getTRTLogger());
  assert(builder == NULL);
  INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  assert(network == NULL);

  int odim = std::stoi(configure["odim"]);
  int nvocab = std::stoi(configure["nvocab"]);
  DataType ctype = configure["trt_dtype"] == "Float" ? DataType::kFLOAT : DataType::kHALF;

  auto input = network->addInput("words", DataType::kINT32, Dims{-1});
  auto encoder = network->addInput("encoder", ctype, DimsHW{ -1,odim });
  auto bottom = PositionalEncoding(network, input, configure);

  bottom = Decoder_Layers(network, bottom, encoder, configure);
  bottom = FinalSlice(network, bottom, "");
  if (configure["normalize_before"] == "True")
    bottom = LayerNormalization(network, bottom, odim, "decoder.after_norm.");

  bottom = FC(network, bottom, odim, nvocab, "decoder.output_layer.");
  auto softmax = network->addSoftMax(*bottom);
  softmax->setAxes(0);

  int top_k = std::stoi(configure["top_k"]);
  auto topk = network->addTopK(*softmax->getOutput(0), TopKOperation::kMAX, top_k, 0);
  auto prob = topk->getOutput(0);
  auto index = topk->getOutput(1);

  prob->setName("prob");
  index->setName("index");
  network->markOutput(*prob);
  network->markOutput(*index);

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
