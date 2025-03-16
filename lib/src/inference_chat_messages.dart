enum FinishReason { stop, unspecified }

class ChatResult {
  final ChatMessage message;
  final FinishReason finishReason;

  ChatResult({required this.message, required this.finishReason});

  @override
  String toString() {
    return 'ChatResult(message: $message, finishReason: $finishReason)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;

    return other is ChatResult &&
        other.message == message &&
        other.finishReason == finishReason;
  }

  @override
  int get hashCode => message.hashCode ^ finishReason.hashCode;
}

class ChatMessage {
  final String role;
  final String content;

  ChatMessage.custom({required this.content, required this.role})
    : assert(role.isNotEmpty);

  ChatMessage.system({required String content})
    : this.custom(content: content, role: 'system');

  ChatMessage.human({required String content})
    : this.custom(content: content, role: 'user');

  ChatMessage.assistant({required String content})
    : this.custom(content: content, role: 'assistant');

  ChatMessage.tool({required String content})
    : this.custom(content: content, role: 'tool');

  @override
  String toString() {
    return 'ChatMessage(role: $role, content: $content)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;

    return other is ChatMessage &&
        other.role == role &&
        other.content == content;
  }

  @override
  int get hashCode => role.hashCode ^ content.hashCode;
}
