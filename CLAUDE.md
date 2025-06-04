# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Building and Testing
- `dart pub get` - Install dependencies
- `dart analyze` - Run static analysis
- `dart test` - Run tests (if any exist)

### Example Usage
- Run the main example: `dart run example/bin/example.dart`
- Run the RAG example: `dart run example/bin/rag.dart`

## Architecture Overview

This is a Dart/Flutter package that provides a high-level interface for running large language model inference using llama.cpp as the backend. The package is structured as follows:

### Core Components

**InferenceModel** (`lib/src/model.dart`):
- Represents a GGUF model file with metadata fetching capabilities
- Handles model loading and metadata extraction without full initialization
- Used before creating an engine to inspect model properties

**InferenceEngine** (`lib/src/engine.dart`):
- Main inference engine that manages the model lifecycle
- Handles model loading, context management, and resource cleanup
- Provides high-level methods for chat, embeddings, tokenization
- Uses finalizers for automatic memory management
- Context is lazily created and reused based on required length

**ChatMessage** (`lib/src/chat_messages.dart`):
- Structured message format for conversational AI
- Supports system, human, and assistant roles
- Used with the chat template system

**Low-level Interface** (`lib/src/low_level.dart`):
- FFI bindings to llama.cpp via llama_cpp_bindings package
- Provides configuration for dynamic library path and logging
- Global singleton for accessing native functions

### Key Design Patterns

**Resource Management**: 
- Uses Dart finalizers to ensure native resources are cleaned up
- Context reuse optimization to avoid recreating contexts unnecessarily
- Explicit dispose() methods for deterministic cleanup

**Streaming Architecture**:
- Chat inference streams results via onResult callback
- Tokens are generated and yielded one at a time for real-time response
- ChatResult objects contain individual pieces or finish reasons

**Memory Optimization**:
- Context size is dynamically allocated based on requirements
- Embedding and chat contexts are created with appropriate configurations
- Token batching for efficient processing

## Dependencies

- **llama_cpp_bindings**: Git dependency providing FFI bindings to llama.cpp
- **ffi**: Dart's foreign function interface package
- **lints**: Standard Dart linting rules

## Native Library Requirements

The package requires llama.cpp to be compiled and available as a dynamic library:
- macOS: `.dylib` file
- Linux: `.so` file  
- Windows: `.dll` file

The library path must be configured via `lowLevelInference.dynamicLibrary` before use.

## Model Format

Only GGUF format models are supported. The package can handle:
- Text generation models
- Embedding models
- Models with chat templates

## Memory Considerations

Models are loaded entirely into RAM. The package provides utilities to estimate memory requirements based on model size, context length, and KV cache needs.