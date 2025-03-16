import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:inference/inference.dart';

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
  Inference? inference;

  @override
  void dispose() {
    inference?.dispose();
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
                if (inference == null) {
                  final picked = await FilePicker.platform.pickFiles(
                    type: FileType.custom,
                    allowedExtensions: ['gguf'],
                    allowMultiple: false,
                  );

                  final modelPath = picked?.files.first.path;

                  if (modelPath == null) return;

                  inference = Inference(modelPath: modelPath);
                }

                await inference!.initialize();

                _output = '';

                final messages = [
                  ChatMessage.human(content: 'Tell me about yourself'),
                ];

                await for (final result in inference!.chat(messages)) {
                  setState(() => _output += result.message.content);
                  await Future.delayed(const Duration());
                }

                inference!.dispose();
              },
              child: Text('Compute'),
            ),
          ],
        ),
      ),
    );
  }
}
