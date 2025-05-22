import 'dart:ffi';
import 'dart:io';
import 'dart:developer';

import 'package:ffi/ffi.dart';
import 'package:llama_cpp_bindings/llama_cpp_bindings.dart';

final lowLevelInference = _LowLevelInference();

typedef LogHandlerCallback = Function(String message);

LogHandlerCallback? _staticLogHandler = log;

// Static callback function that routes to the appropriate instance
void _ffiStaticLogHandler(
  int level,
  Pointer<Char> text,
  Pointer<Void> userData,
) {
  if (_staticLogHandler == null) return;
  final message = text.cast<Utf8>().toDartString().trim();
  if (message.isEmpty) return;
  _staticLogHandler!(message);
}

class _LowLevelInference {
  LlamaBindings? _bindings;

  set logCallback(LogHandlerCallback? callback) {
    _staticLogHandler = callback;

    // Set the static log callback in llama.cpp
    bindings.llama_log_set(
      Pointer.fromFunction<ggml_log_callbackFunction>(_ffiStaticLogHandler),
      nullptr,
    );

    // Set the static log callback in ggml.cpp
    bindings.ggml_log_set(
      Pointer.fromFunction<ggml_log_callbackFunction>(_ffiStaticLogHandler),
      nullptr,
    );
  }

  set dynamicLibrary(DynamicLibrary dynamicLibrary) {
    _bindings = LlamaBindings(dynamicLibrary);
    logCallback = _staticLogHandler;
  }

  LlamaBindings get bindings {
    if (_bindings != null) return _bindings!;

    dynamicLibrary = switch (Platform.operatingSystem) {
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

    return _bindings!;
  }
}
