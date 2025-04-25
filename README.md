# Inference for Dart and Flutter

Run large language model inference from Dart and Flutter, using [`llama.cpp`](https://github.com/ggml-org/llama.cpp) as a backend. The API is designed to be human-friendly and to follow the [Dart design guidelines](https://dart.dev/effective-dart/design).

> [!WARNING]  
> This is a work in progress and not ready for production use. The API is subject to change.

## Installation

1. Add the following to your `pubspec.yaml`:
    ```yaml
    dependencies:
        inference:
            git:
            url: https://github.com/breitburg/inference-for-dart
    ```

2. Run `flutter pub get`.

3. Compile the [`llama.cpp`](https://github.com/ggml-org/llama.cpp) backend for your target platform and link it to your native project.

    _For iOS you need to open the Xcode project and add `llama.xcframework` to the 'Frameworks, Libraries, and Embedded Content' section, for Linux you need to compile `libllama.so` and link it to your project, etc._

## Usage

Run the inference from any `.gguf` model using the following sample code:

```dart
import 'package:inference/inference.dart';

void main() async {
    // Create a model object from the path to the model file
    final model = InferenceModel(path: 'your_model.gguf');

    // Fetch and print the metadata of the model (e.g. name, flavor, etc.)
    final metadata = model.fetchMetadata();
    print('${metadata.name} by ${metadata.organization} under ${metadata.license}');

    // Create an inference engine using the model
    final engine = InferenceEngine(model);

    // Load the model into memory, create context, vocabulary, etc.
    await engine.initialize();

    // Define the messages to send to the model
    final messages = [
        ChatMessage.system(
            'You are a helpful assistant running on ${Platform.operatingSystem}.'
        ),
        ChatMessage.human('Why is the sky blue?'),
    ];

    // Run the inference and print the output parts
    await for (final part in engine.chat(messages)) {
        print(part.message.content);
    }

    // Offload the model from memory and free resources
    engine.dispose();
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Llama.cpp is a project by [GGML](https://ggml.ai/), and this project is built on top of it. Special thanks to:

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for the C++ backend.
- [i-Naji/llama_cpp_dart](https://github.com/i-Naji/llama_cpp_dart) for the automated Dart bindings.
