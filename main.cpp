
#include "Espnet_TRT_Transformer_Encoder.h"
#include "Espnet_TRT_Transformer_Decoder.h"

#ifndef TEST

#include "plugin_registery.h"

#endif

int main(int argc,char**argv) {
#ifndef TEST
  registryFinalSlicePlugin();
  registryLayerNormalizaitonPlugin();
  registryPositionWisePlugin();
  registrySelfAttentionPlugin();
  registrySrcAttentionPlugin();
#endif 

  if (argc < 3) {
    std::cout
      << "--path [the transformer model weight path, Required!]" << std::endl
      << "--idim [input feature dimension, default 83]" << std::endl
      << "--n_Head [the number of head in attention, default 4]" << std::endl
      << "--odim [feature dimension, default 256]" << std::endl
      << "--feed_forward [feed forward dimension, default 2048]" << std::endl
      << "--nvocab [the size of vocabulary, Required!]" << std::endl
      << "--dtype [the data type used for computation {float/half}, default float]" << std::endl
      << "--concat_after [concat is used, default false]" << std::endl
      << "--normalize_before [default true]" << std::endl
      << "--encoder_layers [the number of attention in encoder, default 12]" << std::endl
      << "--decoder_layers [the number of attention in decoder, default 6]" << std::endl
      << "--batchsize [the max batchsize of decoder, default 16]" << std::endl
      << "--topk [the topk in each decoder step, default 16]" << std::endl
      << "--maxseql [the max sequence length of encoder, default 500]" << std::endl
      << "--model_name [the output trt model name, default asr]" << std::endl;
  }
  std::map<std::string, std::string> configure{
    {"--path","asr"},
    {"--idim","83"},
    {"--n_Head","4"},
    {"--odim","256"},
    {"--feed_forward","2048"},
    {"--nvocab","7244"},
    {"--dtype","float"},
    {"--concat_after","false"},
    {"--normalize_before","true"},
    {"--encoder_layers","12"},
    {"--decoder_layers","6"},
    {"--batchsize","16"},
    {"--topk","16"},
    {"--maxseql","5000"},
    {"--model_name","asr"}
  };

  for (int i = 1; i < argc; i += 2) {
    if (configure.count(argv[i]) > 0) {
      configure[argv[i]] = argv[i + 1];
    }
    else {
      std::cerr << "Option is not supported!" << std::endl;
      exit(0);
    }
  }

  if (configure["--path"] == "") {
    std::cerr << "The path to model weight is not specified!" << std::endl;
    std::cout << "To see more information by running program without option input!" << std::endl;
    exit(0);
  }

  if (configure["--nvocab"] == "") {
    std::cerr << "The size of vocabulary is not specified!" << std::endl;
    std::cout << "To see more information by running program without option input!" << std::endl;
  }

  std::cout << "building encoder model ..." << std::endl;
  Espnet_TRT_Transformer_Encoder(configure);

  std::cout << "building decoder model ..." << std::endl;
  Espnet_TRT_Transformer_Decoder(configure);

  return 0;

}