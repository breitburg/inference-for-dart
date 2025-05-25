import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:inference/src/low_level.dart';
import 'package:inference/src/model.dart';
import 'package:llama_cpp_bindings/llama_cpp_bindings.dart';

/// A custom sampler function type that can be applied to token data
typedef CustomSamplerFunction = void Function(Pointer<llama_token_data_array> tokenData);

/// A wrapper for custom samplers to interface with llama.cpp
class CustomInferenceSampler {
  final CustomSamplerFunction _applyFunc;
  Pointer<llama_sampler>? _sampler;

  CustomSamplerFunction get applyFunc => _applyFunc;

  CustomInferenceSampler(this._applyFunc);

  Pointer<llama_sampler> getSampler() {
    if (_sampler != null) return _sampler!;
    
    // Note: This is a simplified version. In a real implementation,
    // you'd need to create proper C callbacks for custom samplers.
    // For now, we'll throw an error as custom samplers require
    // more complex FFI callback setup.
    throw UnimplementedError(
      'Custom samplers require C callback implementation. '
      'Use built-in samplers instead.'
    );
  }

  void dispose() {
    if (_sampler != null) {
      lowLevelInference.bindings.llama_sampler_free(_sampler!);
      _sampler = null;
    }
  }
}

/// A configurable sampler chain for llama.cpp inference
class InferenceSampler {
  Pointer<llama_sampler>? _sampler;
  final List<Pointer<llama_sampler>> _samplers = [];
  final List<CustomInferenceSampler> _customSamplers = [];

  bool get isInitialized => _sampler != null;

  InferenceSampler() {
    final params = lowLevelInference.bindings.llama_sampler_chain_default_params();
    _sampler = lowLevelInference.bindings.llama_sampler_chain_init(params);
    
    if (_sampler == nullptr) {
      throw Exception('Failed to initialize sampler chain');
    }
  }

  /// Add a greedy sampler (always picks the most likely token)
  void addGreedy() {
    final sampler = lowLevelInference.bindings.llama_sampler_init_greedy();
    _addSampler(sampler);
  }

  /// Add a distribution sampler with the given seed
  void addDist(int seed) {
    final sampler = lowLevelInference.bindings.llama_sampler_init_dist(seed);
    _addSampler(sampler);
  }

  /// Add a softmax sampler
  void addSoftmax() {
    final sampler = lowLevelInference.bindings.llama_sampler_init_softmax();
    _addSampler(sampler);
  }

  /// Add a top-k sampler (limits to k most likely tokens)
  void addTopK(int k) {
    final sampler = lowLevelInference.bindings.llama_sampler_init_top_k(k);
    _addSampler(sampler);
  }

  /// Add a top-p sampler (nucleus sampling)
  void addTopP(double p, int minKeep) {
    final sampler = lowLevelInference.bindings.llama_sampler_init_top_p(p, minKeep);
    _addSampler(sampler);
  }

  /// Add a min-p sampler
  void addMinP(double p, int minKeep) {
    final sampler = lowLevelInference.bindings.llama_sampler_init_min_p(p, minKeep);
    _addSampler(sampler);
  }

  /// Add a typical sampler
  void addTypical(double p, int minKeep) {
    final sampler = lowLevelInference.bindings.llama_sampler_init_typical(p, minKeep);
    _addSampler(sampler);
  }

  /// Add a temperature sampler
  void addTemp(double temp) {
    final sampler = lowLevelInference.bindings.llama_sampler_init_temp(temp);
    _addSampler(sampler);
  }

  /// Add an extended temperature sampler
  void addTempExt(double t, double delta, double exponent) {
    final sampler = lowLevelInference.bindings.llama_sampler_init_temp_ext(t, delta, exponent);
    _addSampler(sampler);
  }

  /// Add a Mirostat sampler
  void addMirostat(int nVocab, int seed, double tau, double eta, int m) {
    final sampler = lowLevelInference.bindings.llama_sampler_init_mirostat(
      nVocab, seed, tau, eta, m
    );
    _addSampler(sampler);
  }

  /// Add a Mirostat v2 sampler
  void addMirostatV2(int seed, double tau, double eta) {
    final sampler = lowLevelInference.bindings.llama_sampler_init_mirostat_v2(seed, tau, eta);
    _addSampler(sampler);
  }

  /// Add a grammar sampler
  void addGrammar(InferenceModel model, String grammar, String root) {
    // Note: This requires the model's vocabulary to be accessible
    // You might need to modify this based on your model implementation
    final grammarUtf8 = grammar.toNativeUtf8().cast<Char>();
    final rootUtf8 = root.toNativeUtf8().cast<Char>();
    
    try {
      // This is a simplified version - you'll need to get the vocab from the model
      throw UnimplementedError(
        'Grammar sampler requires model vocabulary access. '
        'This needs to be implemented based on your model structure.'
      );
    } finally {
      malloc.free(grammarUtf8);
      malloc.free(rootUtf8);
    }
  }

  /// Add a penalties sampler
  void addPenalties({
    required int penaltyLastN,
    required double penaltyRepeat,
    required double penaltyFreq,
    required double penaltyPresent,
  }) {
    final sampler = lowLevelInference.bindings.llama_sampler_init_penalties(
      penaltyLastN,
      penaltyRepeat,
      penaltyFreq,
      penaltyPresent,
    );
    _addSampler(sampler);
  }

  /// Initialize a logit bias sampler
  void initLogitBias(int nVocab, int nLogitBias, Pointer<llama_logit_bias> logitBias) {
    final sampler = lowLevelInference.bindings.llama_sampler_init_logit_bias(
      nVocab, nLogitBias, logitBias
    );
    _addSampler(sampler);
  }

  /// Add a Mirostat sampler (requires vocabulary size to be provided)
  void addMirostatWithVocabSize(int nVocab, int seed, double tau, double eta, int m) {
    final sampler = lowLevelInference.bindings.llama_sampler_init_mirostat(
      nVocab, seed, tau, eta, m
    );
    _addSampler(sampler);
  }

  /// Add a custom sampler function
  void addCustom(CustomSamplerFunction applyFunc) {
    final customSampler = CustomInferenceSampler(applyFunc);
    final sampler = customSampler.getSampler();
    _addSampler(sampler);
    _customSamplers.add(customSampler);
  }

  /// Internal method to add a sampler to the chain
  void _addSampler(Pointer<llama_sampler> sampler) {
    if (_sampler == null) {
      throw Exception('Sampler chain not initialized');
    }
    if (sampler == nullptr) {
      throw Exception('Failed to create sampler');
    }
    
    lowLevelInference.bindings.llama_sampler_chain_add(_sampler!, sampler);
    _samplers.add(sampler);
  }

  /// Get the current seed of the sampler
  int getSeed() {
    if (_sampler == null) {
      throw Exception('Sampler chain not initialized');
    }
    return lowLevelInference.bindings.llama_sampler_get_seed(_sampler!);
  }

  /// Sample a token from the given context
  int sample(Pointer<llama_context> context, int idx) {
    if (_sampler == null) {
      throw Exception('Sampler chain not initialized');
    }
    if (context == nullptr) {
      throw Exception('Context is null');
    }
    
    return lowLevelInference.bindings.llama_sampler_sample(_sampler!, context, idx);
  }

  /// Create a default sampler configuration
  static InferenceSampler createDefault({
    double temperature = 0.8,
    double topP = 0.95,
    double topK = 40,
    double minP = 0.05,
    double typicalP = 1.0,
    int seed = LLAMA_DEFAULT_SEED,
    double repeatPenalty = 1.1,
    double frequencyPenalty = 0.0,
    double presencePenalty = 0.0,
    int repeatPenaltyTokens = 64,
  }) {
    final sampler = InferenceSampler();
    
    // Add penalties if needed
    if (repeatPenalty > 1.0 || frequencyPenalty != 0.0 || presencePenalty != 0.0) {
      sampler.addPenalties(
        penaltyLastN: repeatPenaltyTokens,
        penaltyRepeat: repeatPenalty,
        penaltyFreq: frequencyPenalty,
        penaltyPresent: presencePenalty,
      );
    }
    
    // Top-K filter
    if (topK > 0) {
      sampler.addTopK(topK.toInt());
    }
    
    // Min-P filter
    if (minP > 0 && minP < 1.0) {
      sampler.addMinP(minP, 1);
    }
    
    // Typical-P filter
    if (typicalP > 0 && typicalP < 1.0) {
      sampler.addTypical(typicalP, 1);
    }
    
    // Top-P filter
    if (topP < 1.0) {
      sampler.addTopP(topP, 1);
    }
    
    // Temperature sampler
    if (temperature > 0) {
      sampler.addTemp(temperature);
    }
    
    // Distribution sampler (should be last)
    sampler.addDist(seed);
    
    return sampler;
  }

  /// Clean up resources
  void dispose() {
    if (_sampler != null) {
      // Remove custom samplers first to prevent llama.cpp from trying to free them
      for (int i = _customSamplers.length - 1; i >= 0; i--) {
        final samplerIndex = lowLevelInference.bindings.llama_sampler_chain_n(_sampler!) - 1;
        lowLevelInference.bindings.llama_sampler_chain_remove(_sampler!, samplerIndex);
        _customSamplers[i].dispose();
      }
      
      lowLevelInference.bindings.llama_sampler_free(_sampler!);
      _sampler = null;
    }
    
    _samplers.clear();
    _customSamplers.clear();
  }
}