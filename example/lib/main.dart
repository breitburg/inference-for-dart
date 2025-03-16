import 'dart:convert';

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
  InferenceModelInformation? info;

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
            if (info != null) ...[
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    info!.baseName,
                    style: Theme.of(context).textTheme.headlineSmall,
                    textAlign: TextAlign.center,
                  ),
                  SizedBox(width: 10),
                  Center(
                    child: Container(
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.black, width: 1.5),
                        borderRadius: BorderRadius.circular(5),
                      ),
                      padding: EdgeInsets.symmetric(vertical: 2, horizontal: 5),
                      child: Text(
                        info!.sizeLabel,
                        style: Theme.of(context).textTheme.labelLarge,
                      ),
                    ),
                  ),
                ],
              ),
              SizedBox(height: 30),
            ],
            if (_output.isNotEmpty) ...[Text(_output), SizedBox(height: 30)],
            ElevatedButton(
              onPressed: () async {
                if (inference == null) {
                  final picked = await FilePicker.platform.pickFiles(
                    type: FileType.any,
                    allowMultiple: false,
                  );

                  final modelPath = picked?.files.first.path;

                  if (modelPath == null) return;

                  inference = Inference(modelPath: modelPath);
                }

                await inference!.initialize();
                setState(() {
                  info = inference!.fetchInformation();
                });

                _output = '';

                final messages = [
                  ChatMessage.system(content: 'You are Lori, a helpful chatbot'),
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
