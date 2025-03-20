import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:inference/src/inference_chat_messages.dart';
import 'package:inference/src/inference_model.dart';
import 'package:inference/src/singletons.dart';
import 'package:llama_cpp_bindings/llama_cpp_bindings.dart';

class InferenceEngine {
  /// The model to use for inference.
  final InferenceModel model;

  /// The number of GPU layers to use for inference.
  final int gpuLayerCount;

  // Runtime objects
  late Pointer<llama_model> _model;
  late Pointer<llama_vocab> _vocabulary;

  // State flags
  bool _initialized = false;
  bool get initialized => _initialized;

  InferenceEngine(this.model, {this.gpuLayerCount = 99});

  /// Initialize the Llama instance, loading the model
  /// and initializing the context.
  Future<void> initialize() async {
    if (_initialized) return;

    // Load all available backends
    llama.ggml_backend_load_all();

    // Initialize the model
    final modelParams =
        llama.llama_model_default_params()..n_gpu_layers = gpuLayerCount;

    // Load the model from the file
    final modelPathUtf8 = model.path.toNativeUtf8().cast<Char>();
    _model = llama.llama_model_load_from_file(modelPathUtf8, modelParams);
    malloc.free(modelPathUtf8);

    if (_model == nullptr) {
      throw Exception('Unable to load model from path: ${model.path}');
    }

    // Get the vocabulary
    _vocabulary = llama.llama_model_get_vocab(_model);

    _initialized = true;
  }

  /// Tokenize a text string into a list of token IDs.
  ///
  /// The `addBos` parameter controls whether a beginning-of-sequence token
  /// should be added to the beginning of the token list.
  Future<List<int>> tokenize(String text, {bool addBos = true}) async {
    if (!_initialized) throw Exception('Engine not initialized');

    final textUtf8 = text.toNativeUtf8().cast<Char>();

    // Get the number of tokens
    final tokenCount =
        -llama.llama_tokenize(
          _vocabulary,
          textUtf8,
          text.length,
          nullptr,
          0,
          addBos,
          true,
        );

    // Allocate space for tokens and tokenize
    final tokens = malloc<Int32>(tokenCount);

    if (llama.llama_tokenize(
          _vocabulary,
          textUtf8,
          text.length,
          tokens,
          tokenCount,
          addBos,
          true,
        ) <
        0) {
      malloc.free(textUtf8);
      malloc.free(tokens);
      throw Exception('Failed to tokenize text');
    }

    // Convert to Dart list
    final result = List<int>.generate(tokenCount, (i) => tokens[i]);

    // Clean up
    malloc.free(textUtf8);
    malloc.free(tokens);

    return result;
  }

  /// Generates a chat response for a list of messages.
  Stream<ChatResult> chat(
    List<ChatMessage> messages, {
    int contextLength = 512,
    double temperature = 1,
    double topP = 1,
    int seed = LLAMA_DEFAULT_SEED,
  }) async* {
    if (!_initialized) throw Exception('Engine not initialized');

    // Initialize the context
    final contextParams =
        llama.llama_context_default_params()
          ..n_ctx = contextLength
          ..n_batch = contextLength;

    final context = llama.llama_init_from_model(_model, contextParams);

    if (context == nullptr) {
      llama.llama_model_free(_model);
      throw Exception('Failed to create the llama context');
    }

    // Initialize the sampler
    final sampler = llama.llama_sampler_chain_init(
      llama.llama_sampler_chain_default_params(),
    );
    llama.llama_sampler_chain_add(
      sampler,
      llama.llama_sampler_init_top_p(topP, 1),
    );
    llama.llama_sampler_chain_add(
      sampler,
      llama.llama_sampler_init_temp(temperature),
    );
    llama.llama_sampler_chain_add(sampler, llama.llama_sampler_init_dist(seed));

    // Get the chat template from the model
    final templatePtr = llama.llama_model_chat_template(_model, nullptr);
    final template = templatePtr.cast<Utf8>().toDartString();

    // Convert messages to llama_chat_message format
    final llamaMessages =
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
    final messagesArray = malloc<llama_chat_message>(llamaMessages.length);
    for (int i = 0; i < llamaMessages.length; i++) {
      messagesArray[i] = llamaMessages[i];
    }

    // Create buffer for the formatted chat
    final bufferSize = contextLength * 4; // Conservative estimate
    final buffer = malloc<Char>(bufferSize);

    // Apply the template
    final formattedLength = llama.llama_chat_apply_template(
      template.toNativeUtf8().cast<Char>(),
      messagesArray,
      llamaMessages.length,
      true,
      buffer,
      bufferSize,
    );

    // Helper function to free llama_chat_message resources
    void freeMessages(
      List<llama_chat_message> messages,
      Pointer<llama_chat_message> array,
    ) {
      for (var msg in messages) {
        malloc.free(msg.role.cast<Utf8>());
        malloc.free(msg.content.cast<Utf8>());
      }
      malloc.free(array);
    }

    if (formattedLength < 0 || formattedLength > bufferSize) {
      freeMessages(llamaMessages, messagesArray);
      malloc.free(buffer);
      throw Exception('Failed to apply chat template');
    }

    // Convert buffer to Dart string
    final formattedChat = buffer.cast<Utf8>().toDartString(
      length: formattedLength,
    );

    // Track the reason for finishing the chat
    FinishReason? finishReason;

    try {
      // Tokenize the formatted chat
      final tokens = await tokenize(formattedChat, addBos: true);

      // Convert to pointer for llama_batch
      final tokensPtr = malloc<Int32>(tokens.length);
      for (int i = 0; i < tokens.length; i++) {
        tokensPtr[i] = tokens[i];
      }

      // Create batch for the tokens
      var batch = llama.llama_batch_get_one(tokensPtr, tokens.length);

      // Skip context space check since llama_kv_self_used_cells is unavailable
      // Process the batch
      if (llama.llama_decode(context, batch) != 0) {
        malloc.free(tokensPtr);
        throw Exception('Failed to decode batch');
      }

      // We don't need the token pointer anymore after processing the batch
      malloc.free(tokensPtr);

      // Sample tokens until we hit a stop condition
      while (true) {
        // Sample the next token
        final newTokenId = llama.llama_sampler_sample(sampler, context, -1);

        // Check if we reached the end of generation
        if (llama.llama_vocab_is_eog(_vocabulary, newTokenId)) {
          finishReason = FinishReason.stop;
          break;
        }

        // Convert token to text
        final pieceBuffer = malloc<Char>(256);
        final pieceLength = llama.llama_token_to_piece(
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

        // Create a ChatResult to yield
        yield ChatResult(
          message: ChatMessage.assistant(piece),
          finishReason: FinishReason.unspecified, // Not finished yet
        );

        // Prepare the next batch with the new token
        final newTokenPtr = malloc<Int32>(1)..value = newTokenId;
        batch = llama.llama_batch_get_one(newTokenPtr, 1);

        // Process the batch
        if (llama.llama_decode(context, batch) != 0) {
          malloc.free(newTokenPtr);
          throw Exception('Failed to decode batch');
        }

        malloc.free(newTokenPtr);
      }
    } finally {
      // Clean up regardless of success or failure
      freeMessages(llamaMessages, messagesArray);
      llama.llama_sampler_free(sampler);
      llama.llama_free(context);
      malloc.free(buffer);
    }

    // Only yield the final result if we haven't encountered an error
    yield ChatResult(
      message: ChatMessage.assistant(''),
      finishReason: finishReason,
    );
  }

  /// Frees all resources associated with the Llama instance,
  /// including the model, context, sampler, and vocabulary.
  /// Frees all resources associated with the Llama instance.
  Future<void> dispose() async {
    if (!_initialized) return;
    llama.llama_model_free(_model);
    _initialized = false;
  }
}
