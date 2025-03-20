import 'dart:ffi';
import 'dart:io';

import 'package:llama_cpp_bindings/llama_cpp_bindings.dart';

final _dynamicLibrary = switch (Platform.operatingSystem) {
  'ios' => DynamicLibrary.open('llama.framework/llama'),
  'macos' => DynamicLibrary.open('llama.framework/llama'),
  'android' => DynamicLibrary.open('libllama.so'),
  'linux' => DynamicLibrary.open('libllama.so'),
  'windows' => DynamicLibrary.open('llama.dll'),
  _ =>
    throw UnsupportedError(
      'Failed to load `llama.cpp` library for \'${Platform.operatingSystem}\'',
    ),
};

final llama = LlamaBindings(_dynamicLibrary);
