import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:inference/src/low_level.dart';

class InferenceModel {
  final String path;

  const InferenceModel(this.path);

  /// Fetches all metadata from the model file.
  /// 
  /// Returns a [Map] containing all key-value pairs found in the model's metadata.
  Map<String, String> fetchMetadata() {
    // Load the model temporarily to access metadata
    final modelParams = lowLevelInference.bindings.llama_model_default_params();
    final modelPathUtf8 = path.toNativeUtf8().cast<Char>();
    final model = lowLevelInference.bindings.llama_model_load_from_file(
      modelPathUtf8,
      modelParams,
    );
    malloc.free(modelPathUtf8);

    if (model == nullptr) {
      throw Exception('Failed to load model from path: $path');
    }

    try {
      final metadata = <String, String>{};
      final metaCount = lowLevelInference.bindings.llama_model_meta_count(model);
      
      int bufferSize = 1024;
      Pointer<Char> buffer = malloc<Char>(bufferSize);
      
      try {
        for (int i = 0; i < metaCount; i++) {
          // Get key - retry with larger buffer if needed
          int keyBytes = lowLevelInference.bindings.llama_model_meta_key_by_index(
            model, i, buffer, bufferSize
          );
          
          if (keyBytes > bufferSize) {
            malloc.free(buffer);
            bufferSize = keyBytes + 1;
            buffer = malloc<Char>(bufferSize);
            keyBytes = lowLevelInference.bindings.llama_model_meta_key_by_index(
              model, i, buffer, bufferSize
            );
          }
          
          if (keyBytes < 0) {
            continue; // Skip invalid entries
          }
          
          final key = buffer.cast<Utf8>().toDartString(length: keyBytes);
          
          // Get value - retry with larger buffer if needed
          int valueBytes = lowLevelInference.bindings.llama_model_meta_val_str_by_index(
            model, i, buffer, bufferSize
          );
          
          if (valueBytes > bufferSize) {
            malloc.free(buffer);
            bufferSize = valueBytes + 1;
            buffer = malloc<Char>(bufferSize);
            valueBytes = lowLevelInference.bindings.llama_model_meta_val_str_by_index(
              model, i, buffer, bufferSize
            );
          }
          
          if (valueBytes < 0) {
            continue; // Skip invalid entries
          }
          
          final value = buffer.cast<Utf8>().toDartString(length: valueBytes);
          metadata[key] = value;
        }
      } finally {
        malloc.free(buffer);
      }
      
      return metadata;
    } finally {
      lowLevelInference.bindings.llama_model_free(model);
    }
  }

  @override
  String toString() {
    return 'InferenceModel(path: $path)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;

    return other is InferenceModel && other.path == path;
  }

  @override
  int get hashCode => path.hashCode;
}