# Inference for Dart and Flutter

Run large language model inference from Dart and Flutter, using [`llama.cpp`](https://github.com/ggml-org/llama.cpp) as a backend. The API is designed to be human-friendly and to follow the [Dart design guidelines](https://dart.dev/effective-dart/design).

> [!WARNING]  
> This is a work in progress and not ready for production use. The API is subject to change. Currently, the API is blocking the main thread, but it will be changed to use isolates in the future.

## Installation

1. Add the following to your `pubspec.yaml`:
    ```yaml
    dependencies:
        inference:
            git:
                url: https://github.com/breitburg/inference-for-dart
    ```

2. Run `flutter pub get`.

3. Compile the [`llama.cpp`](https://github.com/ggml-org/llama.cpp) backend for your target platform and link it to your native project. Use the official [build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for your platform. The library must be compiled with the same architecture as your Dart/Flutter app.

    _For iOS, you need to open the Xcode project and add `llama.xcframework` to the 'Frameworks, Libraries, and Embedded Content' section; for Linux, you need to compile `libllama.so` and link it to your project, etc._

## Prerequisites

### Model Support

You can run inference from any `.gguf` model, downloaded at runtime or embedded within the app. Search for 'GGUF' on HuggingFace to find and download the model file.

For the up-to-date model availability, see [llama.cpp's README](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#text-only).

### RAM Requirements

Running inference with large language models requires sufficient memory resources. The model must be fully loaded into RAM (or VRAM for GPU acceleration) before any inference can begin.

For example, the [OLMo 2 (7B) Instruct model with `Q4_K_S` quantization](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct-GGUF/blob/main/olmo-2-1124-7B-instruct-Q4_K_S.gguf) is 4.25 GB.

To estimate the minimum RAM required for inference, use this formula:

```
Total RAM ≈ Model Weights + KV Cache + Inference Overhead
```

1. **Model Weights**: 4.25 GB × 1.1 ≈ 4.7 GB
   - Quantized models typically require 1.0-1.2× their file size in memory.

2. **KV Cache**: ~100-200 MB (with default 1024 token context)
   - Calculated as: 2 × n_layers × context_length × (n_heads_kv × head_size) × data_type_size
   - Scales linearly with context length.

3. **Inference Overhead**: ~100-200 MB
   - Includes temporary buffers and computational workspace.

Total Estimated RAM: **~5.0-5.1 GB minimum**

*Insufficient RAM will result in application crash during model initialization.*

## Usage

### Models and Metadata

Before running inference and loading the full model weights into memory, you can inspect the model metadata to fetch its name, authors, the license, and understand its capabilities and requirements.

```dart
// Create a model instance from a file path (not loaded yet)
final model = InferenceModel(path: 'path/to/model.gguf');

// Retrieve and display model metadata without loading the model
final metadata = model.fetchMetadata();
print('${metadata.name} by ${metadata.organization} under ${metadata.license}');
```

### Initializing the Inference Engine

The `InferenceEngine` manages the model's lifecycle, including initialization and cleanup.

```dart
// Create an inference engine with the loaded model
final engine = InferenceEngine(model);

// Initialize the engine (loads model into memory, prepares context, etc.)
engine.initialize();

// Dispose of the engine when done to free resources
engine.dispose();
```

> **Tip:** Always dispose of resources (such as the inference engine) when they are no longer needed to avoid memory leaks.

### Performing Chat Inference

Interact with the model using structured chat messages for conversational AI scenarios.

```dart
// Prepare a list of chat messages with roles
final messages = [
    ChatMessage.system('You are a helpful assistant running on ${Platform.operatingSystem}.'),
    ChatMessage.human('Why is the sky blue?'),
];

// Run inference and handle the output
engine.chat(
    messages,
    onResult: (result) => stdout.write(result.message?.content ?? ""),
);
```

### Tokenizing and Detokenizing Text

You can convert text to tokens, and convert tokens back to text as needed for your application and evaluations.

```dart
// Tokenize input text
List<int> tokens = engine.tokenize("Hello, world!");
print("Tokens: $tokens"); // Example output: [1, 15043, 29892, 0]

// Detokenize tokens back to text
String text = engine.detokenize([1, 15043, 29892, 0]);
print("Text: $text"); // Output: "Hello, world!"
```

### Advanced Configuration

Customize library behavior, such as specifying a dynamic library path or handling logs.

```dart
// Set a custom dynamic library and log callback
lowLevelInference
    ..dynamicLibrary = DynamicLibrary.open("path/to/libllama.so") // Use .dylib for macOS, .dll for Windows
    ..logCallback = (String message) => print('[llama.cpp] $message');
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Llama.cpp is a project by [GGML](https://ggml.ai/), and this project is built on top of it. Special thanks to:

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for the C++ backend.
- [i-Naji/llama_cpp_dart (forked to include `ggml.h`)](https://github.com/breitburg/llama_cpp_dart_bindings) for the automated Dart bindings.