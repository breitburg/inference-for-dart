import 'dart:ffi';
import 'dart:io';
import 'dart:math';

import 'package:inference/inference.dart';

void main(List<String> arguments) async {
  lowLevelInference.dynamicLibrary = DynamicLibrary.open(
    'misc/llama.cpp/build/bin/libllama.so',
  );

  final embeddingModel = InferenceModel(
    '../models/all-MiniLM-L6-v2-Q4_K_M.gguf',
  );
  final embeddingEngine = InferenceEngine(embeddingModel)..initialize();

  final docs = [
    'Anthropic was founded in 2021',
    'Ilia Breitburg is 21 years old as of 2025',
    'Georgi Gerganov is the founder of llama.cpp',
    'OpenAI is (not) evil',
  ];

  final store = {
    for (final string in docs) embeddingEngine.embed(string): string,
  };

  final chatModel = InferenceModel(
    '../models/OLMo-2-0425-1B-Instruct-Q4_0.gguf',
  );

  final chatEngine = InferenceEngine(chatModel)..initialize();
  final history = <ChatMessage>[];

  while (true) {
    stdout.write('> ');
    final query = stdin.readLineSync() ?? '';

    if (query.isEmpty) continue;

    final queryEmbedding = embeddingEngine.embed(query);

    final similarities = {
      for (final entry in store.entries)
        cosineSimilarity(queryEmbedding, entry.key): entry.value,
    }.entries.toList();

    similarities
      ..sort((a, b) => a.key < b.key ? 1 : 0)
      ..removeWhere((e) => e.key < 0.5);

    if (similarities.isEmpty) {
      print('Sorry, I was unable to find relevant context for that query.');
      continue;
    }

    history.add(ChatMessage.human(query));

    var response = '';
    chatEngine.chat(
      [
        ChatMessage.system(
          'You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don\'t know the answer, just say that you don\'t know. Use three sentences maximum and keep the answer concise. Retrieved context: ${similarities.map((e) => '"${e.value}"').join(', ')}',
        ),
        ...history,
      ],
      onResult: (i) {
        final delta = i.message?.content ?? '';
        stdout.write(delta);
        response += delta;
      },
    );

    print('');

    history.add(ChatMessage.assistant(response));
  }
}

double cosineSimilarity(List<double> vectorA, List<double> vectorB) {
  if (vectorA.isEmpty || vectorB.isEmpty) {
    throw ArgumentError('Vectors cannot be empty');
  }

  if (vectorA.length != vectorB.length) {
    throw ArgumentError('Vectors must have the same length');
  }

  double dotProduct = 0.0;
  double magnitudeA = 0.0;
  double magnitudeB = 0.0;

  for (int i = 0; i < vectorA.length; i++) {
    dotProduct += vectorA[i] * vectorB[i];
    magnitudeA += vectorA[i] * vectorA[i];
    magnitudeB += vectorB[i] * vectorB[i];
  }

  magnitudeA = sqrt(magnitudeA);
  magnitudeB = sqrt(magnitudeB);

  // Handle zero vectors
  if (magnitudeA == 0.0 || magnitudeB == 0.0) {
    return 0.0;
  }

  return dotProduct / (magnitudeA * magnitudeB);
}
