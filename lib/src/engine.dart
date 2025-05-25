import 'dart:ffi';
import 'dart:math' show sqrt;

import 'package:ffi/ffi.dart';
import 'package:inference/src/chat_messages.dart';
import 'package:inference/src/model.dart';
import 'package:inference/src/low_level.dart';
import 'package:llama_cpp_bindings/llama_cpp_bindings.dart';

class InferenceEngine {
  /// The model to use for inference.
  final InferenceModel model;

  /// The number of GPU layers to use for inference.
  final int gpuLayerCount;

  // Runtime objects
  late Pointer<llama_model> _model;
  late Pointer<llama_vocab> _vocabulary;
  Pointer<llama_context>? _context;

  // Resource management
  final _modelFinalizer = Finalizer<Pointer<llama_model>>((model) {
    lowLevelInference.bindings.llama_model_free(model);
  });

  final _contextFinalizer = Finalizer<Pointer<llama_context>>((context) {
    lowLevelInference.bindings.llama_free(context);
  });

  // State flags
  bool _initialized = false;
  bool get initialized => _initialized;

  InferenceEngine(this.model, {this.gpuLayerCount = 0});

  /// Initialize the Llama instance, loading the model
  /// and initializing the context.
  void initialize() {
    if (_initialized) return;

    // Load all available backends
    lowLevelInference.bindings.ggml_backend_load_all();

    // Initialize the model
    final modelParams =
        lowLevelInference.bindings.llama_model_default_params()
          ..n_gpu_layers = gpuLayerCount;

    // Load the model from the file
    final modelPathUtf8 = model.path.toNativeUtf8().cast<Char>();
    _model = lowLevelInference.bindings.llama_model_load_from_file(
      modelPathUtf8,
      modelParams,
    );
    malloc.free(modelPathUtf8);

    if (_model == nullptr) {
      throw Exception('Unable to load model from path: ${model.path}');
    }

    // Register finalizer to ensure cleanup
    _modelFinalizer.attach(this, _model, detach: this);

    // Get the vocabulary
    _vocabulary = lowLevelInference.bindings.llama_model_get_vocab(_model);

    _initialized = true;
  }

  /// Get a context appropriate for the given length
  /// Reuses existing context if it's large enough
  Pointer<llama_context> _getContext(
    int requiredLength, {
    bool forEmbeddings = false,
  }) {
    if (!_initialized) throw Exception('Engine not initialized');

    // If we already have a context with sufficient capacity, reuse it
    if (_context != null &&
        lowLevelInference.bindings.llama_n_ctx(_context!) >= requiredLength) {
      return _context!;
    }

    // Free existing context if any
    if (_context != null) {
      _contextFinalizer.detach(this);
      lowLevelInference.bindings.llama_free(_context!);
      _context = null;
    }

    // Create a new context
    final contextParams =
        lowLevelInference.bindings.llama_context_default_params()
          ..n_ctx = requiredLength
          ..n_batch = requiredLength;

    if (forEmbeddings) {
      contextParams.embeddings = true;
      contextParams.n_ubatch = requiredLength; // For non-causal models
    }

    final context = lowLevelInference.bindings.llama_init_from_model(
      _model,
      contextParams,
    );

    if (context == nullptr) {
      throw Exception('Failed to create the llama context');
    }

    // Update state and register finalizer
    _context = context;
    _contextFinalizer.attach(this, context, detach: this);

    return context;
  }

  /// Generate embeddings for the given text.
  ///
  /// Returns a normalized vector of embeddings as a [List<double>].
  /// The [contextLength] parameter controls the maximum context size.
  /// If the text is longer than the context, it will be truncated.
  /// The [normalize] parameter controls whether to normalize the embeddings to unit length.
  List<double> embed(
    String text, {
    int contextLength = 2048,
    bool normalize = true,
  }) {
    if (!_initialized) throw Exception('Engine not initialized');
    if (text.isEmpty) throw ArgumentError('Text cannot be empty');

    // Get context with embeddings enabled
    final context = _getContext(contextLength, forEmbeddings: true);

    // Clear KV cache
    lowLevelInference.bindings.llama_kv_self_clear(context);

    // Tokenize the input text - ensure we add special tokens properly
    final tokens = tokenize(text, special: true, parseSpecial: true);

    if (tokens.isEmpty) {
      throw Exception('Tokenization resulted in empty token list');
    }

    // Ensure tokens fit in context
    final maxTokens = lowLevelInference.bindings.llama_n_batch(context);
    final tokensToProcess =
        tokens.length > maxTokens ? tokens.sublist(0, maxTokens) : tokens;

    // Get model embedding dimensions
    final nEmbd = lowLevelInference.bindings.llama_model_n_embd(_model);

    Pointer<Int32>? tokensPtr;

    try {
      // Create token array
      tokensPtr = malloc<Int32>(tokensToProcess.length);
      for (int i = 0; i < tokensToProcess.length; i++) {
        tokensPtr[i] = tokensToProcess[i];
      }

      // Create batch - make sure all tokens have logits enabled for embeddings
      final batch = lowLevelInference.bindings.llama_batch_init(
        tokensToProcess.length,
        0, // embd = 0 for token-based input
        1, // n_seq_max = 1
      );

      // Set up the batch manually to ensure proper logits
      for (int i = 0; i < tokensToProcess.length; i++) {
        batch.token[i] = tokensToProcess[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0; // sequence ID 0
        batch.logits[i] = 1; // Enable logits for all tokens
      }
      batch.n_tokens = tokensToProcess.length;

      // Use encode for embeddings (not decode)
      final encodeResult = lowLevelInference.bindings.llama_encode(
        context,
        batch,
      );
      if (encodeResult < 0) {
        throw Exception('Failed to encode tokens for embedding: $encodeResult');
      }

      // Get embeddings - try sequence embeddings first, then token embeddings
      Pointer<Float> embeddingsPtr = lowLevelInference.bindings
          .llama_get_embeddings_seq(
            context,
            0, // sequence ID 0
          );

      // If sequence embeddings are null, try token embeddings from last token
      if (embeddingsPtr == nullptr) {
        embeddingsPtr = lowLevelInference.bindings.llama_get_embeddings_ith(
          context,
          tokensToProcess.length - 1,
        );
      }

      // If still null, try general embeddings
      if (embeddingsPtr == nullptr) {
        embeddingsPtr = lowLevelInference.bindings.llama_get_embeddings(
          context,
        );
      }

      if (embeddingsPtr == nullptr) {
        throw Exception(
          'Failed to get embeddings - model may not support embeddings',
        );
      }

      // Convert to Dart list
      final embeddings = List<double>.generate(
        nEmbd,
        (i) => embeddingsPtr[i].toDouble(),
      );

      // Clean up batch
      lowLevelInference.bindings.llama_batch_free(batch);

      // Check for NaN or infinite values
      if (embeddings.any((value) => value.isNaN || value.isInfinite)) {
        throw Exception(
          'Embedding contains invalid values - model may not support embeddings',
        );
      }

      // Normalize if requested
      if (normalize) {
        double norm = 0.0;
        for (final value in embeddings) {
          norm += value * value;
        }
        norm = sqrt(norm);

        if (norm > 0.0) {
          for (int i = 0; i < embeddings.length; i++) {
            embeddings[i] /= norm;
          }
        }
      }

      return embeddings;
    } finally {
      // Clean up allocated memory
      if (tokensPtr != null) {
        malloc.free(tokensPtr);
      }
    }
  }

  /// Tokenize a text string into a list of token IDs.
  ///
  /// The `special` parameter controls whether special tokens should be handled.
  /// The `parseSpecial` parameter controls whether to parse special tokens.
  List<int> tokenize(
    String text, {
    bool parseSpecial = true,
    bool special = false,
  }) {
    if (!_initialized) throw Exception('Engine not initialized');

    final textUtf8 = text.toNativeUtf8().cast<Char>();

    // Get the required token count (returns negative count)
    final requiredTokenCount =
        -lowLevelInference.bindings.llama_tokenize(
          _vocabulary,
          textUtf8,
          text.length,
          nullptr,
          0,
          special,
          parseSpecial,
        );

    if (requiredTokenCount <= 0) {
      malloc.free(textUtf8);
      throw Exception('Failed to estimate token count');
    }

    // Allocate token array
    final tokens = malloc<Int32>(requiredTokenCount);

    // Actually tokenize
    final actualTokenCount = lowLevelInference.bindings.llama_tokenize(
      _vocabulary,
      textUtf8,
      text.length,
      tokens,
      requiredTokenCount,
      special,
      parseSpecial,
    );

    if (actualTokenCount < 0) {
      malloc.free(textUtf8);
      malloc.free(tokens);
      throw Exception('Failed to tokenize text');
    }

    // Convert to Dart list
    final result = List<int>.generate(actualTokenCount, (i) => tokens[i]);

    // Clean up
    malloc.free(textUtf8);
    malloc.free(tokens);

    return result;
  }

  /// Convert a list of token IDs back to a text string.
  ///
  /// The `removeSpecial` parameter controls whether special tokens
  /// should be removed from the output text.
  String detokenize(
    List<int> tokens, {
    bool removeSpecial = true,
    bool unparseSpecial = false,
  }) {
    if (!_initialized) throw Exception('Engine not initialized');

    if (tokens.isEmpty) return '';

    // Create token array
    final tokensPtr = malloc<Int32>(tokens.length);
    for (int i = 0; i < tokens.length; i++) {
      tokensPtr[i] = tokens[i];
    }

    // Allocate a buffer for the output text
    // Estimated at 8 bytes per token, which should be enough for most cases
    int bufferSize = tokens.length * 8;
    Pointer<Char> textBuffer = malloc<Char>(bufferSize);

    // Call llama_detokenize with retries if buffer too small
    int maxAttempts = 3;
    int attempts = 0;
    int textLength = -1;

    while (attempts < maxAttempts) {
      textLength = lowLevelInference.bindings.llama_detokenize(
        _vocabulary,
        tokensPtr,
        tokens.length,
        textBuffer,
        bufferSize,
        removeSpecial,
        unparseSpecial,
      );

      if (textLength >= 0) break; // Success

      // Buffer too small, reallocate and retry
      malloc.free(textBuffer);
      bufferSize *= 2; // Double the buffer size
      textBuffer = malloc<Char>(bufferSize);
      attempts++;
    }

    if (textLength < 0) {
      malloc.free(tokensPtr);
      malloc.free(textBuffer);
      throw Exception(
        'Failed to detokenize tokens after $maxAttempts attempts',
      );
    }

    // Convert the buffer to a Dart string
    final result = textBuffer.cast<Utf8>().toDartString(length: textLength);

    // Clean up
    malloc.free(tokensPtr);
    malloc.free(textBuffer);

    return result;
  }

  /// Creates a sampler chain with the specified parameters
  Pointer<llama_sampler> _createSampler({
    double temperature = 0.8,
    double topP = 1.0,
    double topK = 40,
    double minP = 0.05,
    double typicalP = 1.0,
    int seed = LLAMA_DEFAULT_SEED,
    double repeatPenalty = 1.1,
    double frequencyPenalty = 0.0,
    double presencePenalty = 0.0,
    int repeatPenaltyTokens = 64,
  }) {
    final chainParams =
        lowLevelInference.bindings.llama_sampler_chain_default_params();
    final sampler = lowLevelInference.bindings.llama_sampler_chain_init(
      chainParams,
    );

    if (sampler == nullptr) {
      throw Exception('Failed to initialize sampler chain');
    }

    // First add penalties if needed
    if (repeatPenalty > 1.0) {
      final penaltySampler = lowLevelInference.bindings
          .llama_sampler_init_penalties(
            repeatPenaltyTokens,
            repeatPenalty,
            frequencyPenalty,
            presencePenalty,
          );

      if (penaltySampler == nullptr) {
        lowLevelInference.bindings.llama_sampler_free(sampler);
        throw Exception('Failed to create penalty sampler');
      }

      lowLevelInference.bindings.llama_sampler_chain_add(
        sampler,
        penaltySampler,
      );
    }

    // Top-K filter (limit to K most likely tokens)
    if (topK > 0) {
      final topKSampler = lowLevelInference.bindings.llama_sampler_init_top_k(
        topK.toInt(),
      );
      if (topKSampler == nullptr) {
        lowLevelInference.bindings.llama_sampler_free(sampler);
        throw Exception('Failed to create top-k sampler');
      }

      lowLevelInference.bindings.llama_sampler_chain_add(sampler, topKSampler);
    }

    // Min-P filter (remove tokens below threshold)
    if (minP > 0 && minP < 1.0) {
      final minPSampler = lowLevelInference.bindings.llama_sampler_init_min_p(
        minP,
        1,
      );
      if (minPSampler == nullptr) {
        lowLevelInference.bindings.llama_sampler_free(sampler);
        throw Exception('Failed to create min-p sampler');
      }

      lowLevelInference.bindings.llama_sampler_chain_add(sampler, minPSampler);
    }

    // Typical-P filter (nucleus diversity)
    if (typicalP > 0 && typicalP < 1.0) {
      final typicalPSampler = lowLevelInference.bindings
          .llama_sampler_init_typical(typicalP, 1);
      if (typicalPSampler == nullptr) {
        lowLevelInference.bindings.llama_sampler_free(sampler);
        throw Exception('Failed to create typical-p sampler');
      }

      lowLevelInference.bindings.llama_sampler_chain_add(
        sampler,
        typicalPSampler,
      );
    }

    // Top-P filter (nucleus sampling)
    if (topP < 1.0) {
      final topPSampler = lowLevelInference.bindings.llama_sampler_init_top_p(
        topP,
        1,
      );
      if (topPSampler == nullptr) {
        lowLevelInference.bindings.llama_sampler_free(sampler);
        throw Exception('Failed to create top-p sampler');
      }

      lowLevelInference.bindings.llama_sampler_chain_add(sampler, topPSampler);
    }

    // Temperature sampler
    if (temperature > 0) {
      final tempSampler = lowLevelInference.bindings.llama_sampler_init_temp(
        temperature,
      );
      if (tempSampler == nullptr) {
        lowLevelInference.bindings.llama_sampler_free(sampler);
        throw Exception('Failed to create temperature sampler');
      }

      lowLevelInference.bindings.llama_sampler_chain_add(sampler, tempSampler);
    }

    // Distribution sampler (final sampler in chain)
    final distSampler = lowLevelInference.bindings.llama_sampler_init_dist(
      seed,
    );
    if (distSampler == nullptr) {
      lowLevelInference.bindings.llama_sampler_free(sampler);
      throw Exception('Failed to create distribution sampler');
    }

    lowLevelInference.bindings.llama_sampler_chain_add(sampler, distSampler);

    return sampler;
  }

  /// Generates a chat response for a list of messages.
  void chat(
    List<ChatMessage> messages, {
    int contextLength = 1024,
    int maxTokens = 1024,
    double temperature = 0.8,
    double topP = 0.95,
    double topK = 40,
    double minP = 0.05,
    double typicalP = 1.0,
    double repeatPenalty = 1.1,
    int repeatPenaltyTokens = 64,
    int seed = LLAMA_DEFAULT_SEED,
    Function(ChatResult result)? onResult,
  }) {
    if (!_initialized) throw Exception('Engine not initialized');

    // Get or create appropriate context
    final context = _getContext(contextLength);

    // Clean the KV cache
    lowLevelInference.bindings.llama_kv_self_clear(context);

    // Initialize the sampler
    final sampler = _createSampler(
      temperature: temperature,
      topP: topP,
      topK: topK,
      minP: minP,
      typicalP: typicalP,
      repeatPenalty: repeatPenalty,
      repeatPenaltyTokens: repeatPenaltyTokens,
      seed: seed,
    );

    // Resources to clean up
    Pointer<Char>? buffer;
    Pointer<llama_chat_message>? messagesArray;
    List<llama_chat_message> llamaMessages = [];

    try {
      // Get the chat template from the model
      final templatePtr = lowLevelInference.bindings.llama_model_chat_template(
        _model,
        nullptr,
      );

      if (templatePtr == nullptr) {
        throw Exception('Model does not have a chat template');
      }

      final template = templatePtr.cast<Utf8>().toDartString();

      // Convert messages to llama_chat_message format
      llamaMessages =
          messages.map((msg) {
            final roleUtf8 = msg.role.toNativeUtf8().cast<Char>();
            final contentUtf8 = msg.content.toNativeUtf8().cast<Char>();

            final chatMessage =
                malloc<llama_chat_message>().ref
                  ..role = roleUtf8
                  ..content = contentUtf8;

            return chatMessage;
          }).toList();

      // Create native array of llama_chat_message
      messagesArray = malloc<llama_chat_message>(llamaMessages.length);
      for (int i = 0; i < llamaMessages.length; i++) {
        messagesArray[i] = llamaMessages[i];
      }

      // Create buffer for the formatted chat
      final bufferSize = contextLength * 8; // Conservative estimate
      buffer = malloc<Char>(bufferSize);

      // Apply the template
      final formattedLength = lowLevelInference.bindings
          .llama_chat_apply_template(
            template.toNativeUtf8().cast<Char>(),
            messagesArray,
            llamaMessages.length,
            true, // add assistant
            buffer,
            bufferSize,
          );

      if (formattedLength < 0 || formattedLength > bufferSize) {
        throw Exception(
          'Failed to apply chat template: buffer too small or error',
        );
      }

      // Convert buffer to Dart string
      final formattedChat = buffer.cast<Utf8>().toDartString(
        length: formattedLength,
      );

      // Tokenize the formatted chat
      final tokens = tokenize(formattedChat);

      // Process in batches of up to n_batch tokens
      final maxBatchSize = lowLevelInference.bindings.llama_n_batch(context);
      for (int i = 0; i < tokens.length; i += maxBatchSize) {
        final batchSize =
            i + maxBatchSize > tokens.length ? tokens.length - i : maxBatchSize;

        // Create a batch for this chunk of tokens
        final batchTokens = malloc<Int32>(batchSize);
        for (int j = 0; j < batchSize; j++) {
          batchTokens[j] = tokens[i + j];
        }

        final batch = lowLevelInference.bindings.llama_batch_get_one(
          batchTokens,
          batchSize,
        );

        // Process the batch
        if (lowLevelInference.bindings.llama_decode(context, batch) != 0) {
          malloc.free(batchTokens);
          throw Exception('Failed to decode batch at position $i');
        }

        malloc.free(batchTokens);
      }

      // Sample tokens until we hit a stop condition or the limit
      for (var i = 0; i < maxTokens; i++) {
        // Sample the next token
        final newTokenId = lowLevelInference.bindings.llama_sampler_sample(
          sampler,
          context,
          -1,
        );

        // Check if we reached the end of generation
        if (lowLevelInference.bindings.llama_vocab_is_eog(
          _vocabulary,
          newTokenId,
        )) {
          break;
        }

        // Convert token to text
        final pieceBuffer = malloc<Char>(256);
        final pieceLength = lowLevelInference.bindings.llama_token_to_piece(
          _vocabulary,
          newTokenId,
          pieceBuffer,
          256,
          0,
          true,
        );

        if (pieceLength < 0) {
          malloc.free(pieceBuffer);
          throw Exception('Failed to convert token to text');
        }

        final piece = pieceBuffer.cast<Utf8>().toDartString(
          length: pieceLength,
        );
        malloc.free(pieceBuffer);

        // Create a ChatResult to yield just the new piece
        onResult?.call(ChatResult(message: ChatMessage.assistant(piece)));

        // Prepare a new batch with the new token
        final newTokenPtr = malloc<Int32>(1);
        newTokenPtr[0] = newTokenId;

        final batch = lowLevelInference.bindings.llama_batch_get_one(
          newTokenPtr,
          1,
        );

        // Process the batch
        if (lowLevelInference.bindings.llama_decode(context, batch) != 0) {
          malloc.free(newTokenPtr);
          throw Exception('Failed to decode token batch');
        }

        malloc.free(newTokenPtr);
      }

      // Final result with stop reason - empty message since we already streamed the content
      onResult?.call(ChatResult(finishReason: FinishReason.stop));
    } catch (e) {
      // In case of error, yield a result with the error
      onResult?.call(ChatResult(finishReason: FinishReason.error));
    } finally {
      // Free all allocated resources
      lowLevelInference.bindings.llama_sampler_free(sampler);

      // Free messages
      if (messagesArray != null) {
        for (var msg in llamaMessages) {
          malloc.free(msg.role.cast<Utf8>());
          malloc.free(msg.content.cast<Utf8>());
        }
        malloc.free(messagesArray);
      }

      // Free buffer
      if (buffer != null) {
        malloc.free(buffer);
      }
    }
  }

  /// Frees all resources associated with the Llama instance.
  void dispose() {
    if (!_initialized) return;

    // Free context if exists
    if (_context != null) {
      _contextFinalizer.detach(this);
      lowLevelInference.bindings.llama_free(_context!);
      _context = null;
    }

    // Free model
    _modelFinalizer.detach(this);
    lowLevelInference.bindings.llama_model_free(_model);

    _initialized = false;
  }
}
