import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:langchain_llama_cpp/langchain_llama_cpp.dart';

export 'package:llama_cpp_bindings/llama_cpp_bindings.dart';
export 'package:langchain/langchain.dart';

class Llama {
  // Model parameters
  final String modelPath;
  final DynamicLibrary dynamicLibrary;
  final int gpuLayerCount;

  // Runtime objects
  late LlamaBindings _bindings;
  late Pointer<llama_model> _model;
  late Pointer<llama_vocab> _vocabulary;

  // State flags
  bool _initialized = false;
  bool get initialized => _initialized;

  Llama({
    required this.modelPath,
    required this.dynamicLibrary,
    this.gpuLayerCount = 99,
  });

  // Initialize the Llama instance, loading the model
  // and initializing the context.
  Future<void> initialize() async {
    if (_initialized) return;

    // Load the dynamic library
    _bindings = LlamaBindings(dynamicLibrary);

    // Load all available backends
    _bindings.ggml_backend_load_all();

    // Initialize the model
    final modelParams =
        _bindings.llama_model_default_params()..n_gpu_layers = gpuLayerCount;

    final modelPathUtf8 = modelPath.toNativeUtf8().cast<Char>();
    _model = _bindings.llama_model_load_from_file(modelPathUtf8, modelParams);
    malloc.free(modelPathUtf8);

    if (_model == nullptr) {
      throw Exception('Unable to load model from path: $modelPath');
    }

    // Get the vocabulary
    _vocabulary = _bindings.llama_model_get_vocab(_model);

    _initialized = true;
  }

  /// Tokenize a text string into a list of token IDs.
  ///
  /// The `addBos` parameter controls whether a beginning-of-sequence token
  /// should be added to the beginning of the token list.
  List<int> tokenize(String text, {bool addBos = true}) {
    if (!_initialized) throw Exception('Llama not initialized');

    final textUtf8 = text.toNativeUtf8().cast<Char>();

    // Get the number of tokens
    final tokenCount =
        -_bindings.llama_tokenize(
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

    if (_bindings.llama_tokenize(
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
    double temperature = 0.8,
    double topP = 0.05,
    int topK = 0,
    int seed = LLAMA_DEFAULT_SEED,
  }) async* {
    if (!_initialized) throw Exception('Llama not initialized');

    // Initialize the context
    final contextParams =
        _bindings.llama_context_default_params()
          ..n_ctx = contextLength
          ..n_batch = contextLength;

    final context = _bindings.llama_init_from_model(_model, contextParams);

    if (context == nullptr) {
      _bindings.llama_model_free(_model);
      throw Exception('Failed to create the llama context');
    }

    // Initialize the sampler
    final sampler = _bindings.llama_sampler_chain_init(
      _bindings.llama_sampler_chain_default_params(),
    );
    _bindings.llama_sampler_chain_add(
      sampler,
      _bindings.llama_sampler_init_top_p(topP, 1),
    );
    _bindings.llama_sampler_chain_add(
      sampler,
      _bindings.llama_sampler_init_temp(temperature),
    );
    _bindings.llama_sampler_chain_add(
      sampler,
      _bindings.llama_sampler_init_dist(seed),
    );
    _bindings.llama_sampler_chain_add(
      sampler,
      _bindings.llama_sampler_init_top_k(topK),
    );

    // Get the chat template from the model
    final templatePtr = _bindings.llama_model_chat_template(_model, nullptr);
    final template = templatePtr.cast<Utf8>().toDartString();

    // Convert messages to llama_chat_message format
    final llamaMessages =
        messages.map((msg) {
          final role = _getRoleString(msg);
          final content = msg.contentAsString;

          final roleUtf8 = role.toNativeUtf8().cast<Char>();
          final contentUtf8 = content.toNativeUtf8().cast<Char>();

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
    final formattedLength = _bindings.llama_chat_apply_template(
      template.toNativeUtf8().cast<Char>(),
      messagesArray,
      llamaMessages.length,
      true,
      buffer,
      bufferSize,
    );

    if (formattedLength < 0 || formattedLength > bufferSize) {
      _freeMessages(llamaMessages, messagesArray);
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
      final tokens = tokenize(formattedChat, addBos: true);

      // Convert to pointer for llama_batch
      final tokensPtr = malloc<Int32>(tokens.length);
      for (int i = 0; i < tokens.length; i++) {
        tokensPtr[i] = tokens[i];
      }

      // Create batch for the tokens
      var batch = _bindings.llama_batch_get_one(tokensPtr, tokens.length);

      // Skip context space check since llama_kv_self_used_cells is unavailable
      // Process the batch
      if (_bindings.llama_decode(context, batch) != 0) {
        malloc.free(tokensPtr);
        throw Exception('Failed to decode batch');
      }

      // We don't need the token pointer anymore after processing the batch
      malloc.free(tokensPtr);

      // Sample tokens until we hit a stop condition
      while (true) {
        // Sample the next token
        final newTokenId = _bindings.llama_sampler_sample(sampler, context, -1);

        // Check if we reached the end of generation
        if (_bindings.llama_vocab_is_eog(_vocabulary, newTokenId)) {
          finishReason = FinishReason.stop;
          break;
        }

        // Convert token to text
        final pieceBuffer = malloc<Char>(256);
        final pieceLength = _bindings.llama_token_to_piece(
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
          id: DateTime.now().millisecondsSinceEpoch.toString(),
          output: AIChatMessage(content: piece),
          finishReason: FinishReason.unspecified, // Not finished yet
          metadata: {},
          usage: const LanguageModelUsage(),
          streaming: true,
        );

        // Prepare the next batch with the new token
        final newTokenPtr = malloc<Int32>(1)..value = newTokenId;
        batch = _bindings.llama_batch_get_one(newTokenPtr, 1);

        // Skip the context space check here as well

        // Process the batch
        if (_bindings.llama_decode(context, batch) != 0) {
          malloc.free(newTokenPtr);
          throw Exception('Failed to decode batch');
        }

        malloc.free(newTokenPtr);
      }
    } catch (e) {
      // Yield the final result with the error in metadata
      yield ChatResult(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        output: AIChatMessage(content: ''),
        finishReason: finishReason ?? FinishReason.unspecified,
        metadata: {'error': e.toString()},
        usage: const LanguageModelUsage(),
        streaming: false,
      );
    } finally {
      // Clean up regardless of success or failure
      _freeMessages(llamaMessages, messagesArray);
      _bindings.llama_sampler_free(sampler);
      _bindings.llama_free(context);
      malloc.free(buffer);
    }

    // Only yield the final result if we haven't encountered an error
    if (finishReason != null) {
      yield ChatResult(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        output: AIChatMessage(content: ''),
        finishReason: finishReason,
        metadata: {},
        usage: const LanguageModelUsage(),
        streaming: false,
      );
    }
  }

  // Helper function to free llama_chat_message resources
  void _freeMessages(
    List<llama_chat_message> messages,
    Pointer<llama_chat_message> array,
  ) {
    for (var msg in messages) {
      malloc.free(msg.role.cast<Utf8>());
      malloc.free(msg.content.cast<Utf8>());
    }
    malloc.free(array);
  }

  // Helper to get role string from ChatMessage
  String _getRoleString(ChatMessage message) {
    return switch (message) {
      final SystemChatMessage _ => 'system',
      final HumanChatMessage _ => 'user',
      final AIChatMessage _ => 'assistant',
      final ToolChatMessage _ => 'tool',
      final CustomChatMessage custom => custom.role,
    };
  }

  /// Frees all resources associated with the Llama instance,
  /// including the model, context, sampler, and vocabulary.
  /// Frees all resources associated with the Llama instance.
  void dispose() {
    if (!_initialized) return;
    _bindings.llama_model_free(_model);
    _initialized = false;
  }
}
