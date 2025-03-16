import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:llama_cpp_bindings/llama_cpp_bindings.dart';

final _llama = LlamaBindings(switch (Platform.operatingSystem) {
  'ios' => DynamicLibrary.open('llama.framework/llama'),
  'macos' => DynamicLibrary.open('llama.framework/llama'),
  'android' => DynamicLibrary.open('libllama.so'),
  'linux' => DynamicLibrary.open('libllama.so'),
  'windows' => DynamicLibrary.open('llama.dll'),
  _ =>
    throw UnsupportedError(
      'Failed to load `llama.cpp` library for \'${Platform.operatingSystem}\'',
    ),
});

enum FinishReason { stop, unspecified }

class InferenceModel {
  final String path;

  const InferenceModel(this.path);

  InferenceModelMetadata fetchMetadata() {
    final params =
        malloc<gguf_init_params>().ref
          ..no_alloc = false
          ..ctx = nullptr;

    final modelPathUtf8 = path.toNativeUtf8().cast<Char>();
    final ctx = _llama.gguf_init_from_file(modelPathUtf8, params);
    malloc.free(modelPathUtf8);

    if (ctx == nullptr) {
      throw Exception('Failed to load model from path: $path');
    }

    String? getStringValueOrNull(String key) {
      final keyUtf8 = key.toNativeUtf8().cast<Char>();
      final keyId = _llama.gguf_find_key(ctx, keyUtf8);
      malloc.free(keyUtf8);

      if (keyId == -1) return null;

      final value = _llama.gguf_get_val_str(ctx, keyId);
      return value.cast<Utf8>().toDartString();
    }

    try {
      return InferenceModelMetadata.raw(
        name: getStringValueOrNull('general.name'),
        organization: getStringValueOrNull('general.organization'),
        architecture: getStringValueOrNull('general.architecture'),
        type: getStringValueOrNull('general.type'),
        finetune: getStringValueOrNull('general.finetune'),
        baseName: getStringValueOrNull('general.basename'),
        sizeLabel: getStringValueOrNull('general.size_label'),
        license: getStringValueOrNull('general.license'),
        licenseLink: getStringValueOrNull('general.license.link'),
        repoUrl: getStringValueOrNull('general.repo_url'),
      );
    } finally {
      _llama.gguf_free(ctx);
    }
  }

  @override
  String toString() {
    return 'InferenceModel(path: $path)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;

    return other is InferenceModel && other.path == path;
  }

  @override
  int get hashCode => path.hashCode;
}

class InferenceModelMetadata {
  final String? name;
  final String? organization;
  final String? architecture;
  final String? type;
  final String? finetune;
  final String? baseName;
  final String? sizeLabel;
  final String? license;
  final String? licenseLink;
  final String? repoUrl;

  InferenceModelMetadata.raw({
    this.name,
    this.organization,
    this.architecture,
    this.type,
    this.finetune,
    this.baseName,
    this.sizeLabel,
    this.license,
    this.licenseLink,
    this.repoUrl,
  });

  @override
  String toString() {
    return 'InferenceModelInformation('
        'name: $name, '
        'organization: $organization, '
        'architecture: $architecture, '
        'type: $type, '
        'finetune: $finetune, '
        'baseName: $baseName, '
        'sizeLabel: $sizeLabel, '
        'license: $license, '
        'licenseLink: $licenseLink, '
        'repoUrl: $repoUrl'
        ')';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;

    return other is InferenceModelMetadata &&
        other.name == name &&
        other.organization == organization &&
        other.architecture == architecture &&
        other.type == type &&
        other.finetune == finetune &&
        other.baseName == baseName &&
        other.sizeLabel == sizeLabel &&
        other.license == license &&
        other.licenseLink == licenseLink &&
        other.repoUrl == repoUrl;
  }

  @override
  int get hashCode {
    return name.hashCode ^
        organization.hashCode ^
        architecture.hashCode ^
        type.hashCode ^
        finetune.hashCode ^
        baseName.hashCode ^
        sizeLabel.hashCode ^
        license.hashCode ^
        licenseLink.hashCode ^
        repoUrl.hashCode;
  }
}

class ChatResult {
  final ChatMessage message;
  final FinishReason finishReason;

  ChatResult({required this.message, required this.finishReason});

  @override
  String toString() {
    return 'ChatResult(message: $message, finishReason: $finishReason)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;

    return other is ChatResult &&
        other.message == message &&
        other.finishReason == finishReason;
  }

  @override
  int get hashCode => message.hashCode ^ finishReason.hashCode;
}

class ChatMessage {
  final String role;
  final String content;

  ChatMessage.custom({required this.content, required this.role})
    : assert(role.isNotEmpty);

  ChatMessage.system({required String content})
    : this.custom(content: content, role: 'system');

  ChatMessage.human({required String content})
    : this.custom(content: content, role: 'user');

  ChatMessage.assistant({required String content})
    : this.custom(content: content, role: 'assistant');

  ChatMessage.tool({required String content})
    : this.custom(content: content, role: 'tool');

  @override
  String toString() {
    return 'ChatMessage(role: $role, content: $content)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;

    return other is ChatMessage &&
        other.role == role &&
        other.content == content;
  }

  @override
  int get hashCode => role.hashCode ^ content.hashCode;
}

class InferenceEngine {
  // Model parameters
  final InferenceModel model;

  /// The number of GPU layers to use for inference.
  ///
  /// The default value is 99.
  /// If you want to use CPU only, set this to 0.
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
    _llama.ggml_backend_load_all();

    // Initialize the model
    final modelParams =
        _llama.llama_model_default_params()..n_gpu_layers = gpuLayerCount;

    // Load the model from the file
    final modelPathUtf8 = model.path.toNativeUtf8().cast<Char>();
    _model = _llama.llama_model_load_from_file(modelPathUtf8, modelParams);
    malloc.free(modelPathUtf8);

    if (_model == nullptr) {
      throw Exception('Unable to load model from path: ${model.path}');
    }

    // Get the vocabulary
    _vocabulary = _llama.llama_model_get_vocab(_model);

    _initialized = true;
  }

  /// Tokenize a text string into a list of token IDs.
  ///
  /// The `addBos` parameter controls whether a beginning-of-sequence token
  /// should be added to the beginning of the token list.
  List<int> tokenize(String text, {bool addBos = true}) {
    if (!_initialized) throw Exception('Engine not initialized');

    final textUtf8 = text.toNativeUtf8().cast<Char>();

    // Get the number of tokens
    final tokenCount =
        -_llama.llama_tokenize(
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

    if (_llama.llama_tokenize(
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
        _llama.llama_context_default_params()
          ..n_ctx = contextLength
          ..n_batch = contextLength;

    final context = _llama.llama_init_from_model(_model, contextParams);

    if (context == nullptr) {
      _llama.llama_model_free(_model);
      throw Exception('Failed to create the llama context');
    }

    // Initialize the sampler
    final sampler = _llama.llama_sampler_chain_init(
      _llama.llama_sampler_chain_default_params(),
    );
    _llama.llama_sampler_chain_add(
      sampler,
      _llama.llama_sampler_init_top_p(topP, 1),
    );
    _llama.llama_sampler_chain_add(
      sampler,
      _llama.llama_sampler_init_temp(temperature),
    );
    _llama.llama_sampler_chain_add(
      sampler,
      _llama.llama_sampler_init_dist(seed),
    );

    // Get the chat template from the model
    final templatePtr = _llama.llama_model_chat_template(_model, nullptr);
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
    final formattedLength = _llama.llama_chat_apply_template(
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
      final tokens = tokenize(formattedChat, addBos: true);

      // Convert to pointer for llama_batch
      final tokensPtr = malloc<Int32>(tokens.length);
      for (int i = 0; i < tokens.length; i++) {
        tokensPtr[i] = tokens[i];
      }

      // Create batch for the tokens
      var batch = _llama.llama_batch_get_one(tokensPtr, tokens.length);

      // Skip context space check since llama_kv_self_used_cells is unavailable
      // Process the batch
      if (_llama.llama_decode(context, batch) != 0) {
        malloc.free(tokensPtr);
        throw Exception('Failed to decode batch');
      }

      // We don't need the token pointer anymore after processing the batch
      malloc.free(tokensPtr);

      // Sample tokens until we hit a stop condition
      while (true) {
        // Sample the next token
        final newTokenId = _llama.llama_sampler_sample(sampler, context, -1);

        // Check if we reached the end of generation
        if (_llama.llama_vocab_is_eog(_vocabulary, newTokenId)) {
          finishReason = FinishReason.stop;
          break;
        }

        // Convert token to text
        final pieceBuffer = malloc<Char>(256);
        final pieceLength = _llama.llama_token_to_piece(
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
          message: ChatMessage.assistant(content: piece),
          finishReason: FinishReason.unspecified, // Not finished yet
        );

        // Prepare the next batch with the new token
        final newTokenPtr = malloc<Int32>(1)..value = newTokenId;
        batch = _llama.llama_batch_get_one(newTokenPtr, 1);

        // Skip the context space check here as well

        // Process the batch
        if (_llama.llama_decode(context, batch) != 0) {
          malloc.free(newTokenPtr);
          throw Exception('Failed to decode batch');
        }

        malloc.free(newTokenPtr);
      }
    } catch (e) {
      rethrow;
    } finally {
      // Clean up regardless of success or failure
      freeMessages(llamaMessages, messagesArray);
      _llama.llama_sampler_free(sampler);
      _llama.llama_free(context);
      malloc.free(buffer);
    }

    // Only yield the final result if we haven't encountered an error
    yield ChatResult(
      message: ChatMessage.assistant(content: ''),
      finishReason: finishReason,
    );
  }

  /// Frees all resources associated with the Llama instance,
  /// including the model, context, sampler, and vocabulary.
  /// Frees all resources associated with the Llama instance.
  void dispose() {
    if (!_initialized) return;
    _llama.llama_model_free(_model);
    _initialized = false;
  }
}
