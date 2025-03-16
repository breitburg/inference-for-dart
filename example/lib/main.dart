import 'dart:io';

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
    return MaterialApp(
      title: 'Flutter Demo',
      home: HomeScreen(),
      theme: ThemeData(
        colorScheme: ColorScheme.fromSwatch(primarySwatch: Colors.grey),
      ),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _output = '';
  InferenceModelMetadata? info;
  InferenceEngine? engine;

  @override
  void dispose() {
    engine?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: ListView(
        padding: const EdgeInsets.all(30) + MediaQuery.of(context).padding,
        children: [
          if (info != null) ...[
            Row(
              children: [
                Text(
                  info!.baseName ?? 'No Name',
                  style: Theme.of(context).textTheme.headlineSmall,
                ),

                SizedBox(width: 10),
                Center(
                  child: Container(
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.black, width: 1.5),
                      borderRadius: BorderRadius.circular(5),
                    ),
                    padding: EdgeInsets.symmetric(horizontal: 5),
                    child: Text(
                      info!.sizeLabel ?? 'Unknown',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
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
              if (engine == null) {
                final picked = await FilePicker.platform.pickFiles(
                  type: FileType.any,
                  allowMultiple: false,
                );

                final modelPath = picked?.files.first.path;

                if (modelPath == null) return;

                final model = InferenceModel(modelPath);

                setState(() => info = model.fetchMetadata());

                await Future.delayed(const Duration(milliseconds: 50));

                engine = InferenceEngine(model);
              }

              await engine!.initialize();

              _output = '';

              final messages = [
                ChatMessage.system(
                  content:
                      'You are a helpful assistant based on ${engine!.model.fetchMetadata().baseName} model. You\'re running on ${Platform.operatingSystem}. It\'s currently ${DateTime.now().toIso8601String()}.',
                ),
                ChatMessage.human(
                  content:
                      'What do you know about the environment you\'re running in and your identity?',
                ),
              ];

              await for (final result in engine!.chat(messages)) {
                setState(() => _output += result.message.content);
                await Future.delayed(const Duration());
              }
            },
            child: Text('Compute'),
          ),
        ],
      ),
    );
  }
}
