import 'dart:ffi';
import 'dart:io';

import 'package:inference/inference.dart';

void main(List<String> arguments) async {
  lowLevelInference
    ..dynamicLibrary = DynamicLibrary.open(
      "/Users/breitburg/Developer/llama.cpp/build/bin/libllama.dylib",
    );

  final model = InferenceModel(
    '/Users/breitburg/Library/Mobile Documents/com~apple~CloudDocs/AI Models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf',
  );

  final metadata = model.fetchMetadata();

  print(
    '${metadata.name} from ${metadata.organization} under ${metadata.license}',
  );

  final engine = InferenceEngine(model);
  engine.initialize();

  final history = [ChatMessage.human('Why is the sky blue?')];

  engine.chat(
    history,
    onResult: (result) => stdout.write(result.message?.content ?? ""),
  );

  engine.dispose();
}
