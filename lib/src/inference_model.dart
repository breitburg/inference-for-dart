
import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:inference/src/singletons.dart';
import 'package:llama_cpp_bindings/llama_cpp_bindings.dart';

class InferenceModel {
  final String path;

  const InferenceModel(this.path);

  InferenceModelMetadata fetchMetadata() {
    final params =
        malloc<gguf_init_params>().ref
          ..no_alloc = false
          ..ctx = nullptr;

    final modelPathUtf8 = path.toNativeUtf8().cast<Char>();
    final ctx = llama.gguf_init_from_file(modelPathUtf8, params);
    malloc.free(modelPathUtf8);

    if (ctx == nullptr) {
      throw Exception('Failed to load model from path: $path');
    }

    String? getStringValueOrNull(String key) {
      final keyUtf8 = key.toNativeUtf8().cast<Char>();
      final keyId = llama.gguf_find_key(ctx, keyUtf8);
      malloc.free(keyUtf8);

      if (keyId == -1) return null;

      final value = llama.gguf_get_val_str(ctx, keyId);
      return value.cast<Utf8>().toDartString();
    }

    try {
      return InferenceModelMetadata.raw(
        name: getStringValueOrNull('general.name'),
        organization: getStringValueOrNull('general.organization'),
        architecture: getStringValueOrNull('general.architecture'),
        type: getStringValueOrNull('general.type'),
        finetune: getStringValueOrNull('general.finetune'),
        baseName: getStringValueOrNull('general.basename'),
        sizeLabel: getStringValueOrNull('general.size_label'),
        license: getStringValueOrNull('general.license'),
        licenseLink: getStringValueOrNull('general.license.link'),
        repoUrl: getStringValueOrNull('general.repo_url'),
      );
    } finally {
      llama.gguf_free(ctx);
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

class InferenceModelMetadata {
  final String? name;
  final String? organization;
  final String? architecture;
  final String? type;
  final String? finetune;
  final String? baseName;
  final String? sizeLabel;
  final String? license;
  final String? licenseLink;
  final String? repoUrl;

  InferenceModelMetadata.raw({
    this.name,
    this.organization,
    this.architecture,
    this.type,
    this.finetune,
    this.baseName,
    this.sizeLabel,
    this.license,
    this.licenseLink,
    this.repoUrl,
  });

  @override
  String toString() {
    return 'InferenceModelInformation('
        'name: $name, '
        'organization: $organization, '
        'architecture: $architecture, '
        'type: $type, '
        'finetune: $finetune, '
        'baseName: $baseName, '
        'sizeLabel: $sizeLabel, '
        'license: $license, '
        'licenseLink: $licenseLink, '
        'repoUrl: $repoUrl'
        ')';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;

    return other is InferenceModelMetadata &&
        other.name == name &&
        other.organization == organization &&
        other.architecture == architecture &&
        other.type == type &&
        other.finetune == finetune &&
        other.baseName == baseName &&
        other.sizeLabel == sizeLabel &&
        other.license == license &&
        other.licenseLink == licenseLink &&
        other.repoUrl == repoUrl;
  }

  @override
  int get hashCode {
    return name.hashCode ^
        organization.hashCode ^
        architecture.hashCode ^
        type.hashCode ^
        finetune.hashCode ^
        baseName.hashCode ^
        sizeLabel.hashCode ^
        license.hashCode ^
        licenseLink.hashCode ^
        repoUrl.hashCode;
  }
}