import 'dart:ffi';
import 'dart:io';

import 'package:inference/inference.dart';

void main(List<String> arguments) async {
  lowLevelInference
    ..dynamicLibrary = DynamicLibrary.open(
      "misc/llama.cpp/build/bin/libllama.so",
    )
    ..logCallback = (String message) => print('[llama.cpp] $message');

  final model = InferenceModel("../models/OLMo-2-0425-1B-Instruct-Q4_0.gguf");
  final metadata = model.fetchMetadata();

  print(
    'Loaded ${metadata.name} from ${metadata.organization} under ${metadata.license}',
  );

  final engine = InferenceEngine(model);
  await engine.initialize();

  final history = [
    ChatMessage.system('You are a helpful assistant name Dart.'),
  ];

  while (true) {
    stdout.write('> ');
    final input = stdin.readLineSync()!;
    history.add(ChatMessage.human(input));

    var content = '';
    await for (final part in engine.chat(history)) {
      content += part.message.content;
      stdout.write(part.message.content);
    }

    history.add(ChatMessage.assistant(content));

    stdout.writeln();
  }
}
