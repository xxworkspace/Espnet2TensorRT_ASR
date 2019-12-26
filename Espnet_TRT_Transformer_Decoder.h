#pragma once

#include <map>
#include "util.h"
#include <vector>
#include <cassert>
#include "logger.h"
#include <NvInfer.h>

using namespace nvinfer1;

ITensor* PositionalEncoding(
  INetworkDefinition* network,ITensor* input,
  std::map<std::string, std::string> configure) {
  int odim = std::stoi(configure["--odim"]);
  int nvocab = std::stoi(configure["--nvocab"]);

  Weights embed = getWeight(configure["--path"] + "/decoder.embed.0.weight", nvocab * odim * sizeof(float));
  auto embedding = network->addConstant(DimsHW(nvocab, odim), embed)->getOutput(0);
  auto gather = network->addGather(*embedding, *input, 0)->getOutput(0);
  //logTensorInfo(gather,"embedding");

  DataType ctype = configure["--dtype"] == "float" ? DataType::kFLOAT : DataType::kHALF;
  auto bottom = PositionWise(network, gather, std::stoi(configure["--maxseql"]),
    odim, ctype, configure["--path"] + "/decoder.embed.1.pe");
  //logTensorInfo(gather, "PositionWise");

  return bottom;
}

ITensor* Decoder_Layer(
  INetworkDefinition *network,
  ITensor* input,
  ITensor* encoder,
  std::string prefix,
  std::map<std::string, std::string> configure) {
  DataType ctype = configure["--dtype"] == "float" ? DataType::kFLOAT : DataType::kHALF;
  bool normalize_before = configure["--normalize_before"] == "true";
  bool concat_after = configure["--concat_after"] == "true";
  int feed_forward = std::stoi(configure["--feed_forward"]);
  int n_head = std::stoi(configure["--n_Head"]);
  int odim = std::stoi(configure["--odim"]);

  auto tmp = input;
  if (normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + ".norm1");

  auto self = SelfAttention(network, tmp, n_head, odim, 1, prefix);
  if (concat_after) {
    std::vector<ITensor*> vit{ tmp, self };
    auto concat = network->addConcatenation(vit.data(), vit.size())->getOutput(0);
    auto fc = FC(network, concat, odim * 2, odim, prefix + ".concat_linear1");
    tmp = network->addElementWise(*input,*fc,ElementWiseOperation::kSUM)->getOutput(0);
  }
  else
    tmp = network->addElementWise(*input,*self, ElementWiseOperation::kSUM)->getOutput(0);

  if (!normalize_before)
    tmp = LayerNormalization(network, tmp, odim,prefix + ".norm1");

  input = tmp;
  if (normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + ".norm2");

  auto src = SrcAttention(network, tmp, encoder, n_head, odim, ctype, prefix);

  if (concat_after) {
    std::vector<ITensor*> vit{ tmp, src };
    auto concat = network->addConcatenation(vit.data(), vit.size())->getOutput(0);
    auto fc = FC(network, concat, odim * 2, odim, prefix + ".concat_linear2");
    tmp = network->addElementWise(*input, *fc, ElementWiseOperation::kSUM)->getOutput(0);
  }else
    tmp = network->addElementWise(*input, *src, ElementWiseOperation::kSUM)->getOutput(0);

  if (!normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + ".norm2");

  input = tmp;
  if (normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + ".norm3");

  tmp = FeedForward(network, tmp, odim, feed_forward, prefix);
  tmp = network->addElementWise(*input, *tmp, ElementWiseOperation::kSUM)->getOutput(0);
  if (!normalize_before)
    tmp = LayerNormalization(network, tmp, odim, prefix + ".norm3");

  return tmp;
}

ITensor* Decoder_Layers(
  INetworkDefinition *network,
  ITensor* input,ITensor* encoder,
  std::map<std::string, std::string> configure) {

  int layers = std::stoi(configure["--decoder_layers"]);
  for (int i = 0; i < layers; ++i)
    input = Decoder_Layer(network,input, encoder,configure["--path"] + "/decoder.decoders." + std::to_string(i),configure);

  return input;
}

void Espnet_TRT_Transformer_Decoder(
  std::map<std::string, std::string> configure) {

  Logger logger;
  IBuilder* builder = createInferBuilder(logger.getTRTLogger());
  assert(builder != NULL);
  INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  assert(network != NULL);

  int odim = std::stoi(configure["--odim"]);
  int nvocab = std::stoi(configure["--nvocab"]);
  DataType ctype = configure["--dtype"] == "float" ? DataType::kFLOAT : DataType::kHALF;

  auto input = network->addInput("words", DataType::kINT32, Dims{ 1,-1 });
  auto encoder = network->addInput("encoder", ctype, DimsHW(-1, odim));
  auto bottom = PositionalEncoding(network, input, configure);

  bottom = Decoder_Layers(network, bottom, encoder, configure);
  bottom = FinalSlice(network, bottom, "");
  auto shuffle = network->addShuffle(*bottom);
  shuffle->setReshapeDimensions(DimsHW(1, odim));
  if (configure["--normalize_before"] == "true")
    bottom = LayerNormalization(network, shuffle->getOutput(0), odim, configure["--path"] + "/decoder.after_norm");

  bottom = FC(network, bottom, odim, nvocab, configure["--path"] + "/decoder.output_layer");
  auto softmax = network->addSoftMax(*bottom);
  softmax->setAxes(2);

  int top_k = std::stoi(configure["--topk"]);
  auto topk = network->addTopK(*softmax->getOutput(0), TopKOperation::kMAX, top_k, 2);
  auto prob = topk->getOutput(0);
  auto index = topk->getOutput(1);

  prob->setName("prob");
  index->setName("index");
  network->markOutput(*prob);
  network->markOutput(*index);

  IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("words", OptProfileSelector::kMIN, Dims{ 1, 1 });
  profile->setDimensions("words", OptProfileSelector::kOPT, Dims{ 1, 64 });
  profile->setDimensions("words", OptProfileSelector::kMAX, Dims{ 1, 192 });

  profile->setDimensions("encoder", OptProfileSelector::kMIN, DimsHW(100, 256));
  profile->setDimensions("encoder", OptProfileSelector::kOPT, DimsHW(500, 256));
  profile->setDimensions("encoder", OptProfileSelector::kMAX, DimsHW(1600, 256));

  IBuilderConfig* config = builder->createBuilderConfig();
  config->addOptimizationProfile(profile);
  config->setMaxWorkspaceSize(1 < 30);
  if (configure["--dtype"] == "half")
    config->setFlag(BuilderFlag::kFP16);

  builder->setMaxBatchSize(std::stoi(configure["--batchsize"]));
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  IHostMemory* model = engine->serialize();
  std::ofstream ofs(configure["--model_name"] + "_decoder_" + configure["--dtype"] + ".trt", std::ios::binary);
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
