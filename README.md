# Inference for Dart and Flutter

Run large language models inference natively in Dart and Flutter, using [`llama.cpp`](https://github.com/ggml-org/llama.cpp) as a backend.

> **Note:** This is a work in progress and is not yet ready for production use. API is subject to change.

## Installation

1. Add the following to your `pubspec.yaml`:
    ```yaml
    dependencies:
    inference:
        git:
        url: https://github.com/breitburg/inference
    ```

2. Run `flutter pub get`.

3. Compile the [`llama.cpp`](https://github.com/ggml-org/llama.cpp) backend for your target platform and link it to your native project. For example, in iOS you need to open the Xcode project and add `llama.xcframework` to the `Frameworks, Libraries, and Embedded Content` section.

## Usage

Run the inference using the following code:

```dart
import 'package:inference/inference.dart';

void main() {
  final inference = Inference(
        // You need to provide the path to the model file
        modelPath: 'your_model_path.gguf',

        // Optional: You need to provide the path to the compiled `llama.cpp` backend for your target platform
        dynamicLibrary: DynamicLibrary.open('llama.framework/llama'),
    );

    // Initialize the inference and load the model into memory
    await inference!.initialize();

    // Define the messages to send to the model
    final messages = [
        ChatMessage.human(content: 'Why is the sky blue?'),
    ];

    // Run the inference and print the output
    await for (final result in inference!.chat(messages)) {
        print(result.message.content);
    }

    // Offload the model from memory
    inference!.dispose();
}
```
