import 'dart:ffi';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:langchain_llama_cpp/langchain_llama_cpp.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(title: 'Flutter Demo', home: HomeScreen());
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _output = '';
  Llama? llama;

  @override
  void dispose() {
    llama?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(30),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(child: SingleChildScrollView(child: Text(_output))),
            SizedBox(height: 30),
            ElevatedButton(
              onPressed: () async {
                if (llama == null) {
                  final picked = await FilePicker.platform.pickFiles(
                    type: FileType.custom,
                    allowedExtensions: ['gguf'],
                    allowMultiple: false,
                  );

                  final modelPath = picked?.files.first.path;

                  if (modelPath == null) return;

                  llama = Llama(
                    modelPath: modelPath,
                    dynamicLibrary: DynamicLibrary.open(
                      'llama.framework/llama',
                    ),
                  );
                }

                await llama!.initialize();

                _output = '';

                final messages = [
                  HumanChatMessage(
                    content: ChatMessageContent.text('Tell me about yourself'),
                  ),
                ];

                await for (final result in llama!.chat(messages)) {
                  setState(() => _output += result.output.content);
                  await Future.delayed(const Duration());
                }

                llama!.dispose();
              },
              child: Text('Click me'),
            ),
          ],
        ),
      ),
    );
  }
}
